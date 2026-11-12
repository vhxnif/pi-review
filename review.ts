/**
 * Code Review Extension (inspired by Codex's review feature)
 *
 * Provides a `/review` command that prompts the agent to review code changes.
 * Supports multiple review modes:
 * - Review a GitHub pull request (checks out the PR locally)
 * - Review against a base branch (PR style)
 * - Review uncommitted changes
 * - Review one or more commits (multi-select in the selector)
 * - Review every commit after a given commit
 * - Shared custom review instructions (applied to all review modes when configured)
 *
 * Usage:
 * - `/review` - show interactive selector
 * - `/review pr 123` - review PR #123 (checks out locally)
 * - `/review pr https://github.com/owner/repo/pull/123` - review PR from URL
 * - `/review uncommitted` - review uncommitted changes directly
 * - `/review branch main` - review against main branch
 * - `/review commit abc123` - review specific commit
 * - `/review commits abc123 def456` - review several commits together
 * - `/review after abc123` - review every commit after abc123 (exclusive)
 * - `/review folder src docs` - review specific folders/files (snapshot, not diff)
 * - `/review` selector includes Add/Remove custom review instructions (applies to all modes)
 * - `/review --extra "focus on performance regressions"` - add extra review instruction (works with any mode)
 * - `/review uncommitted --auto` - review, fix findings, and re-review until clean
 * - `/review branch main --auto --review-model anthropic/claude-opus-4 --max-rounds 4`
 * - `/review-auto-stop` - stop the running auto review loop
 *
 * Project-specific review guidelines:
 * - If a REVIEW_GUIDELINES.md file exists in the same directory as .pi,
 *   its contents are appended to the review prompt.
 *
 * Note: PR review requires a clean working tree (no uncommitted changes to tracked files).
 */

import type {
	ExtensionAPI,
	ExtensionContext,
	ExtensionCommandContext,
	SessionTreeEvent,
} from "@earendil-works/pi-coding-agent";
import { DynamicBorder, BorderedLoader } from "@earendil-works/pi-coding-agent";
import {
	Container,
	fuzzyFilter,
	Input,
	Key,
	matchesKey,
	type SelectItem,
	SelectList,
	Spacer,
	Text,
} from "@earendil-works/pi-tui";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { promises as fs } from "node:fs";

// State to track fresh session review (where we branched from).
// Module-level state means only one review can be active at a time.
// This is intentional - the UI and /end-review command assume a single active review.
let reviewOriginId: string | undefined = undefined;
// Identifies the review branch itself. Reviews can share the same `reviewOriginId`
// (same return target, and nested reviews inherit ancestor state entries), so only
// the branch id written when the review started tells branches apart.
let reviewBranchId: string | undefined = undefined;
let endReviewInProgress = false;
let reviewCustomInstructions: string | undefined = undefined;

const REVIEW_STATE_TYPE = "review-session";
const REVIEW_ANCHOR_TYPE = "review-anchor";
const REVIEW_SETTINGS_TYPE = "review-settings";
const AUTO_STATE_TYPE = "review-auto";
const AUTO_RESULT_PATTERN = /<!--\s*REVIEW_RESULT:\s*([\s\S]*?)\s*-->/g;
const GH_SETUP_INSTRUCTIONS =
	"Install GitHub CLI (`gh`) from https://cli.github.com/ (macOS: `brew install gh`), then sign in with `gh auth login` and verify with `gh auth status`.";
const PR_CHECKOUT_BLOCKED_BY_PENDING_CHANGES_MESSAGE =
	"Cannot checkout PR: you have uncommitted changes. Please commit or stash them first.";

type ReviewSessionState = {
	active: boolean;
	originId?: string;
	// Unique id generated when the review branch is created. `originId` is the
	// return target on the main branch, so nested reviews can share it; this id
	// identifies the review branch itself.
	branchId?: string;
	// Model/thinking level that were active before a review branch took over.
	// /end-review restores them so the main branch keeps its own settings.
	mainModel?: ModelRef;
	mainThinkingLevel?: ReviewThinkingLevel;
	// Review-only model selected for this branch. `/tree` does not replay the
	// branch's model_change entries, so re-entering the branch reapplies it.
	reviewModel?: ModelRef;
};

type ReviewSettingsState = {
	customInstructions?: string;
};

type ReviewModel = NonNullable<ExtensionContext["model"]>;
type ReviewThinkingLevel = ReturnType<ExtensionAPI["getThinkingLevel"]>;

type ModelRef = {
	provider: string;
	id: string;
};

// Auto review loop state. One snapshot is appended per transition, on the branch
// where the transition happens, so branch-local reads stay meaningful.
type AutoPhase = "review" | "fix" | "done" | "stopped" | "error";

type AutoFinding = {
	id: string;
	priority: string;
	location: string;
	title: string;
	detail?: string;
};

type AutoReviewResult = {
	verdict: "pass" | "needs-attention";
	findings: AutoFinding[];
};

type AutoSnapshot = {
	v: 1;
	runId: string;
	branch: "main" | "review";
	phase: AutoPhase;
	round: number;
	maxRounds: number;
	target: ReviewTarget;
	originId: string;
	mainLeafId: string;
	reviewLeafId?: string;
	mainModel?: ModelRef;
	mainThinkingLevel?: ReviewThinkingLevel;
	reviewModel?: ModelRef;
	lastVerdict?: AutoReviewResult["verdict"];
	findingKeys?: string[];
	findingCount?: number;
	lastFixChangedTree?: boolean;
	reason?: string;
	startedAt: string;
	updatedAt: string;
};

type AutoReviewOptions = {
	reviewModel?: ReviewModel;
	maxRounds: number;
	summary?: boolean;
	extraInstruction?: string;
};

// Runtime flags. The persisted snapshot is the recovery record; these flags drive
// the in-process loop and are intentionally not restored automatically.
let autoLoopRunning = false;
let autoStopRequested = false;
let autoRunStarted = false;
let autoRunSettled = false;
let autoRunAborted = false;

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractAssistantText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.map((block) => {
			if (!block || typeof block !== "object") return "";
			const candidate = block as { type?: unknown; text?: unknown };
			return candidate.type === "text" && typeof candidate.text === "string" ? candidate.text : "";
		})
		.filter((text) => text.length > 0)
		.join("\n");
}

function getLastAssistantEntry(
	ctx: ExtensionContext,
): { id: string; text: string; stopReason?: string } | undefined {
	const entries = ctx.sessionManager.getBranch();
	for (let i = entries.length - 1; i >= 0; i--) {
		const entry = entries[i];
		if (entry.type !== "message" || entry.message.role !== "assistant") continue;
		return {
			id: entry.id,
			text: extractAssistantText(entry.message.content),
			stopReason: entry.message.stopReason,
		};
	}
	return undefined;
}

function findingKey(finding: AutoFinding): string {
	return `${finding.location}::${finding.title}`;
}

// Each failure mode is reported separately: unattended runs need to tell a
// missing block apart from malformed JSON and from an unusable findings list.
type AutoReviewParseFailure =
	| "missing-result-block"
	| "invalid-json"
	| "invalid-verdict"
	| "invalid-findings"
	| "needs-attention-without-findings";

type AutoReviewParse =
	| { ok: true; result: AutoReviewResult }
	| { ok: false; reason: AutoReviewParseFailure; detail?: string };

function describeAutoReviewParseFailure(reason: AutoReviewParseFailure): string {
	switch (reason) {
		case "missing-result-block":
			return "no REVIEW_RESULT block was found in the review output";
		case "invalid-json":
			return "the REVIEW_RESULT block is not valid JSON";
		case "invalid-verdict":
			return 'the REVIEW_RESULT "verdict" must be "pass" or "needs-attention"';
		case "invalid-findings":
			return "the REVIEW_RESULT findings are malformed (each needs a location and a title)";
		case "needs-attention-without-findings":
			return 'the REVIEW_RESULT verdict is "needs-attention" but no findings were reported';
	}
}

function parseAutoReviewResult(text: string): AutoReviewParse {
	let lastMatch: string | undefined;
	AUTO_RESULT_PATTERN.lastIndex = 0;
	for (let match = AUTO_RESULT_PATTERN.exec(text); match; match = AUTO_RESULT_PATTERN.exec(text)) {
		lastMatch = match[1];
	}
	if (!lastMatch) return { ok: false, reason: "missing-result-block" };

	let raw: unknown;
	try {
		raw = JSON.parse(lastMatch);
	} catch (error) {
		return {
			reason: "invalid-json",
			detail: error instanceof Error ? error.message : String(error),
			ok: false,
		};
	}
	if (!raw || typeof raw !== "object") {
		return { ok: false, reason: "invalid-json", detail: "the REVIEW_RESULT payload is not a JSON object" };
	}

	const record = raw as { verdict?: unknown; findings?: unknown };
	const verdict = record.verdict === "pass" || record.verdict === "needs-attention" ? record.verdict : undefined;
	if (!verdict) {
		return {
			ok: false,
			reason: "invalid-verdict",
			detail: `got ${JSON.stringify(record.verdict)}`,
		};
	}

	if (record.findings !== undefined && !Array.isArray(record.findings)) {
		return { ok: false, reason: "invalid-findings", detail: '"findings" must be an array' };
	}

	const findings: AutoFinding[] = [];
	for (const [index, item] of (record.findings ?? []).entries()) {
		if (!item || typeof item !== "object") {
			return { ok: false, reason: "invalid-findings", detail: `finding ${index + 1} is not an object` };
		}
		const finding = item as Record<string, unknown>;
		const location =
			typeof finding.location === "string"
				? finding.location.trim()
				: typeof finding.file === "string"
					? finding.file.trim()
					: "";
		const title =
			typeof finding.title === "string"
				? finding.title.trim()
				: typeof finding.summary === "string"
					? finding.summary.trim()
					: "";
		// A finding without a location or title cannot be acted on and would
		// collapse into the same key as every other malformed finding.
		if (!location || !title) {
			return {
				ok: false,
				reason: "invalid-findings",
				detail: `finding ${index + 1} is missing a location or a title`,
			};
		}
		const detail =
			typeof finding.detail === "string"
				? finding.detail.trim()
				: typeof finding.why === "string"
					? finding.why.trim()
					: "";
		findings.push({
			id: String(finding.id ?? `F${index + 1}`),
			priority: String(finding.priority ?? "P2"),
			location,
			title,
			detail: detail ? detail.slice(0, 800) : undefined,
		});
	}

	// needs-attention without a usable findings array is a malformed review, not
	// a pass. Reporting it as a parse failure makes the loop stop.
	if (verdict === "needs-attention" && findings.length === 0) {
		return { ok: false, reason: "needs-attention-without-findings" };
	}

	// A "pass" with findings is contradictory; treat it as needs-attention so we
	// do not silently discard actionable issues.
	return {
		ok: true,
		result: {
			verdict: findings.length === 0 ? "pass" : verdict === "pass" ? "needs-attention" : verdict,
			findings,
		},
	};
}

function formatFindings(findings: AutoFinding[]): string {
	if (findings.length === 0) return "(none)";
	return findings
		.map((finding, index) => {
			const head = `${index + 1}. [${finding.priority}] ${finding.id} — ${finding.location} — ${finding.title}`;
			if (!finding.detail) return head;
			const detail = finding.detail.replace(/\s*\n\s*/g, " ").trim();
			return `${head}\n   ${detail}`;
		})
		.join("\n");
}

function buildAutoProtocol(round: number, mode: "review" | "verify", previousFindings: AutoFinding[]): string {
	const verifyContext =
		mode === "verify"
			? `\nThis is automated verification round ${round}. Fixes were applied to the same working tree.\nRead the current files from disk; do not trust diffs or code snippets from earlier in this conversation.\n\nPrevious findings:\n${formatFindings(previousFindings)}\n\nRe-check every previous finding. Only report findings that are still present. Keep the same ids when possible. Omit fixed findings. If everything is fixed, return \"verdict\":\"pass\" with an empty findings array.\n`
			: "";

	return `## Automated review protocol (required)

This review runs inside an automated fix/verify loop. Do not edit files during a review round.${verifyContext}
End your final message with exactly one machine-readable HTML comment (no code fence, no trailing text after it):

<!--REVIEW_RESULT:{"verdict":"needs-attention","findings":[{"id":"F1","priority":"P1","location":"path/to/file.ext:line","title":"short actionable issue","detail":"1-3 sentences: why it matters and what should change"}]}-->

Rules:
- Use "verdict":"pass" only when there are no actionable findings; findings must then be empty.
- Otherwise use "verdict":"needs-attention" and list every actionable finding.
- priority must be one of P0, P1, P2, P3.
- location must be a concrete path with a line number where possible.
- detail should be short (1-3 sentences); it is what the fix agent receives instead of the full review prose.
- Keep finding ids stable across rounds when reporting the same issue.
- Emit the comment exactly once, at the very end.`;
}

function buildAutoFixPrompt(findings: AutoFinding[]): string {
	return `The automated code review found the following actionable issues. Fix them now in the current working tree.

Hard constraints:
- Do NOT commit, switch git branches, stash, reset, or checkout. Leave changes uncommitted in the shared working tree.
- Keep the scope to these findings.
- Treat "Human Reviewer Callouts (Non-Blocking)" as informational only.

Process:
- Fix in priority order: P0, P1, then P2 (include P3 only if quick and safe).
- If a finding is invalid, already fixed, or not possible right now, briefly explain why and continue.
- Follow fail-fast error handling: do not add local catch/fallback recovery unless this scope is an explicit boundary that can safely translate the failure.
- If you add or keep a try/catch, explain the expected failure mode and either rethrow with context or return a boundary-safe error response.
- Run relevant tests/checks for touched code where practical.
- End with: fixed items, deferred/skipped items (with reasons), and verification results.

Findings:
${formatFindings(findings)}`;
}

function requestAutoStop(ctx: ExtensionContext): void {
	autoStopRequested = true;
	ctx.abort();
	ctx.ui.notify("Stopping the auto review loop...", "info");
}

function setReviewWidget(ctx: ExtensionContext, active: boolean) {
	if (!ctx.hasUI) return;
	if (!active) {
		ctx.ui.setWidget("review", undefined);
		return;
	}

	ctx.ui.setWidget("review", (_tui, theme) => {
		const message = "Review session active, return with /end-review";
		const text = new Text(theme.fg("warning", message), 0, 0);
		return {
			render(width: number) {
				return text.render(width);
			},
			invalidate() {
				text.invalidate();
			},
		};
	});
}

type ReviewBranchState = {
	/** Stable identity of the branch. Falls back to the entry id for legacy states. */
	identity?: string;
	/** Latest review-session entry on the branch. Carries the state that applies here. */
	latestState?: ReviewSessionState;
};

function getReviewBranchState(ctx: ExtensionContext, fromId?: string): ReviewBranchState {
	let latestState: ReviewSessionState | undefined;
	let latestEntryId: string | undefined;
	const branch = fromId === undefined ? ctx.sessionManager.getBranch() : ctx.sessionManager.getBranch(fromId);
	for (const entry of branch) {
		if (entry.type !== "custom" || entry.customType !== REVIEW_STATE_TYPE) continue;
		latestEntryId = entry.id;
		latestState = entry.data as ReviewSessionState;
	}

	return { identity: latestState?.branchId ?? latestEntryId, latestState };
}

function getReviewState(ctx: ExtensionContext, fromId?: string): ReviewSessionState | undefined {
	return getReviewBranchState(ctx, fromId).latestState;
}

/**
 * Latest review state across the whole session, not just the current branch.
 * Review-session entries are appended on the review branch, so they disappear
 * from `getBranch()` as soon as that branch is left.
 */
function getLatestReviewSession(ctx: ExtensionContext): ReviewSessionState | undefined {
	let state: ReviewSessionState | undefined;
	for (const entry of ctx.sessionManager.getEntries()) {
		if (entry.type === "custom" && entry.customType === REVIEW_STATE_TYPE) {
			state = entry.data as ReviewSessionState | undefined;
		}
	}

	return state;
}

function applyReviewState(ctx: ExtensionContext) {
	const branch = getReviewBranchState(ctx);
	const state = branch.latestState;

	if (state?.active && state.originId) {
		reviewOriginId = state.originId;
		reviewBranchId = branch.identity;
		setReviewWidget(ctx, true);
		return;
	}

	reviewOriginId = undefined;
	reviewBranchId = undefined;
	setReviewWidget(ctx, false);
}

function getReviewSettings(ctx: ExtensionContext): ReviewSettingsState {
	let state: ReviewSettingsState | undefined;
	for (const entry of ctx.sessionManager.getEntries()) {
		if (entry.type === "custom" && entry.customType === REVIEW_SETTINGS_TYPE) {
			state = entry.data as ReviewSettingsState | undefined;
		}
	}

	return {
		customInstructions: state?.customInstructions?.trim() || undefined,
	};
}

function applyReviewSettings(ctx: ExtensionContext) {
	const state = getReviewSettings(ctx);
	reviewCustomInstructions = state.customInstructions?.trim() || undefined;
}

// Review target types (matching Codex's approach)
type ReviewCommit = { sha: string; title?: string };

type ReviewTarget =
	| { type: "uncommitted" }
	| { type: "baseBranch"; branch: string }
	| {
			type: "commits";
			commits: ReviewCommit[]; // oldest commit last (git log order)
			/**
			 * Explicit exclusive base for a range review ("every commit after X").
			 * Deriving the base from the oldest selected commit's parent is wrong for
			 * non-linear history, where it can pull in commits outside the range.
			 */
			rangeBase?: string;
		}
	| { type: "pullRequest"; prNumber: number; baseBranch: string; title: string }
	| { type: "folder"; paths: string[] };

// Prompts (adapted from Codex)
const UNCOMMITTED_PROMPT =
	"Review the current code changes (staged, unstaged, and untracked files) and provide prioritized findings.";

const BASE_BRANCH_PROMPT_WITH_MERGE_BASE =
	"Review the code changes against the base branch '{baseBranch}'. The merge base commit for this comparison is {mergeBaseSha}. Run `git diff {mergeBaseSha}` to inspect the changes relative to {baseBranch}. Provide prioritized, actionable findings.";

const BASE_BRANCH_PROMPT_FALLBACK =
	"Review the code changes against the base branch '{branch}'. Start by finding the merge diff between the current branch and {branch}'s upstream e.g. (`git merge-base HEAD \"$(git rev-parse --abbrev-ref \"{branch}@{upstream}\")\"`), then run `git diff` against that SHA to see what changes we would merge into the {branch} branch. Provide prioritized, actionable findings.";

const COMMIT_PROMPT_WITH_TITLE =
	'Review the code changes introduced by commit {sha} ("{title}"). Provide prioritized, actionable findings.';

const COMMIT_PROMPT = "Review the code changes introduced by commit {sha}. Provide prioritized, actionable findings.";

const COMMITS_DIFF_PROMPT =
	"Review the code changes introduced by the following {count} commits (oldest first):\n{commits}\n\nThese commits form a contiguous range, so `git diff {base}...{newest}` shows exactly their combined changes. Run that command to inspect the diff. Provide prioritized, actionable findings.";

const COMMITS_RANGE_PROMPT =
	"Review the code changes introduced by the following {count} commits, i.e. every commit after {base} up to {newest} (oldest first):\n{commits}\n\n`git diff {base}...{newest}` shows exactly their combined changes. Run that command to inspect the diff. Provide prioritized, actionable findings.";

const COMMITS_PROMPT =
	"Review the code changes introduced by the following {count} commits (oldest first):\n{commits}\n\nInspect each commit individually with `git diff <sha>^ <sha>` (for a merge commit this shows the changes merged in) and review the changes across all of them together. Provide prioritized, actionable findings.";

const PULL_REQUEST_PROMPT =
	'Review pull request #{prNumber} ("{title}") against the base branch \'{baseBranch}\'. The merge base commit for this comparison is {mergeBaseSha}. Run `git diff {mergeBaseSha}` to inspect the changes that would be merged. Provide prioritized, actionable findings.';

const PULL_REQUEST_PROMPT_FALLBACK =
	'Review pull request #{prNumber} ("{title}") against the base branch \'{baseBranch}\'. Start by finding the merge base between the current branch and {baseBranch} (e.g., `git merge-base HEAD {baseBranch}`), then run `git diff` against that SHA to see the changes that would be merged. Provide prioritized, actionable findings.';

const FOLDER_REVIEW_PROMPT =
	"Review the code in the following paths: {paths}. This is a snapshot review (not a diff). Read the files directly in these paths and provide prioritized, actionable findings.";

// All review output must be written in Chinese.
const REVIEW_OUTPUT_LANGUAGE =
	"Write the entire review in Chinese (简体中文), including findings, verdict, and callouts. Keep code snippets, file paths, and identifiers unchanged.";

// The detailed review rubric (adapted from Codex's review_prompt.md)
const REVIEW_RUBRIC = `# Review Guidelines

You are acting as a code reviewer for a proposed code change made by another engineer.

Below are default guidelines for determining what to flag. These are not the final word — if you encounter more specific guidelines elsewhere (in a developer message, user message, file, or project review guidelines appended below), those override these general instructions.

## Determining what to flag

Flag issues that:
1. Meaningfully impact the accuracy, performance, security, or maintainability of the code.
2. Are discrete and actionable (not general issues or multiple combined issues).
3. Don't demand rigor inconsistent with the rest of the codebase.
4. Were introduced in the changes being reviewed (not pre-existing bugs).
5. The author would likely fix if aware of them.
6. Don't rely on unstated assumptions about the codebase or author's intent.
7. Have provable impact on other parts of the code — it is not enough to speculate that a change may disrupt another part, you must identify the parts that are provably affected.
8. Are clearly not intentional changes by the author.
9. Be particularly careful with untrusted user input and follow the specific guidelines to review.
10. Treat silent local error recovery (especially parsing/IO/network fallbacks) as high-signal review candidates unless there is explicit boundary-level justification.
11. Violate the clean-code guidelines below.
12. Introduce error handling that conflicts with the fail-fast guidelines below.

## Clean-code guidelines

1. Check whether each newly added function duplicates existing functionality elsewhere in the codebase. Flag actual duplication and identify the existing implementation.
2. Flag one-off helper functions that add indirection without improving clarity or reuse (for example, \`isRecord\` or \`asString\`).
3. Flag abstractions introduced without a concrete need in the reviewed change, including wrappers created only for hypothetical future use.
4. Flag defensive checks or fallback behavior that mask programming errors, especially when callers already guarantee the relevant invariants.

## Untrusted User Input

1. Be careful with open redirects, they must always be checked to only go to trusted domains (?next_page=...)
2. Always flag SQL that is not parametrized
3. In systems with user supplied URL input, http fetches always need to be protected against access to local resources (intercept DNS resolver!)
4. Escape, don't sanitize if you have the option (eg: HTML escaping)

## Comment guidelines

1. Be clear about why the issue is a problem.
2. Communicate severity appropriately - don't exaggerate.
3. Be brief - at most 1 paragraph.
4. Keep code snippets under 3 lines, wrapped in inline code or code blocks.
5. Use \`\`\`suggestion blocks ONLY for concrete replacement code (minimal lines; no commentary inside the block). Preserve the exact leading whitespace of the replaced lines.
6. Explicitly state scenarios/environments where the issue arises.
7. Use a matter-of-fact tone - helpful AI assistant, not accusatory.
8. Write for quick comprehension without close reading.
9. Avoid excessive flattery or unhelpful phrases like "Great job...".

## Review priorities

1. Surface critical non-blocking human callouts (migrations, dependency churn, auth/permissions, compatibility, destructive operations) at the end.
2. Prefer simple, direct solutions over wrappers or abstractions without clear value.
3. Treat back pressure handling as critical to system stability.
4. Apply system-level thinking; flag changes that increase operational risk or on-call wakeups.
5. Ensure that errors are always checked against codes or stable identifiers, never error messages.

## Fail-fast error handling (strict)

When reviewing added or modified error handling, default to fail-fast behavior.

1. Evaluate every new or changed \`try/catch\`: identify what can fail and why local handling is correct at that exact layer.
2. Prefer propagation over local recovery. If the current scope cannot fully recover while preserving correctness, rethrow (optionally with context) instead of returning fallbacks.
3. Flag catch blocks that hide failure signals (e.g. returning \`null\`/\`[]\`/\`false\`, swallowing JSON parse failures, logging-and-continue, or “best effort” silent recovery).
4. JSON parsing/decoding should fail loudly by default. Quiet fallback parsing is only acceptable with an explicit compatibility requirement and clear tested behavior.
5. Boundary handlers (HTTP routes, CLI entrypoints, supervisors) may translate errors, but must not pretend success or silently degrade.
6. If a catch exists only to satisfy lint/style without real handling, treat it as a bug.
7. When uncertain, prefer crashing fast over silent degradation.

## Required human callouts (non-blocking, at the very end)

After findings/verdict, you MUST append this final section:

## Human Reviewer Callouts (Non-Blocking)

Include only applicable callouts (no yes/no lines):

- **This change adds a database migration:** <files/details>
- **This change introduces a new dependency:** <package(s)/details>
- **This change changes a dependency (or the lockfile):** <files/package(s)/details>
- **This change modifies auth/permission behavior:** <what changed and where>
- **This change introduces backwards-incompatible public schema/API/contract changes:** <what changed and where>
- **This change includes irreversible or destructive operations:** <operation and scope>
- **This change adds or removes feature flags:** <feature flags changed> (call out re-use of dormant feature flags!)
- **This change changes configuration defaults:** <config var changed>

Rules for this section:
1. These are informational callouts for the human reviewer, not fix items.
2. Do not include them in Findings unless there is an independent defect.
3. These callouts alone must not change the verdict.
4. Only include callouts that apply to the reviewed change.
5. Keep each emitted callout bold exactly as written.
6. If none apply, write "- (none)".

## Priority levels

Tag each finding with a priority level in the title:
- [P0] - Drop everything to fix. Blocking release/operations. Only for universal issues that do not depend on assumptions about inputs.
- [P1] - Urgent. Should be addressed in the next cycle.
- [P2] - Normal. To be fixed eventually.
- [P3] - Low. Nice to have.

## Output format

Provide your findings in a clear, structured format:
1. List each finding with its priority tag, file location, and explanation.
2. Findings must reference locations that overlap with the actual diff — don't flag pre-existing code.
3. Keep line references as short as possible (avoid ranges over 5-10 lines; pick the most suitable subrange).
4. Provide an overall verdict: "correct" (no blocking issues) or "needs attention" (has blocking issues).
5. Ignore trivial style issues unless they obscure meaning or violate documented standards.
6. Do not generate a full PR fix — only flag issues and optionally provide short suggestion blocks.
7. End with the required "Human Reviewer Callouts (Non-Blocking)" section and all applicable bold callouts (no yes/no).

Output all findings the author would fix if they knew about them. If there are no qualifying findings, explicitly state the code looks good. Don't stop at the first finding - list every qualifying issue. Then append the required non-blocking callouts section.`;

async function loadProjectReviewGuidelines(cwd: string): Promise<string | null> {
	let currentDir = path.resolve(cwd);

	while (true) {
		const piDir = path.join(currentDir, ".pi");
		const guidelinesPath = path.join(currentDir, "REVIEW_GUIDELINES.md");

		const piStats = await fs.stat(piDir).catch(() => null);
		if (piStats?.isDirectory()) {
			const guidelineStats = await fs.stat(guidelinesPath).catch(() => null);
			if (guidelineStats?.isFile()) {
				try {
					const content = await fs.readFile(guidelinesPath, "utf8");
					const trimmed = content.trim();
					return trimmed ? trimmed : null;
				} catch {
					return null;
				}
			}
			return null;
		}

		const parentDir = path.dirname(currentDir);
		if (parentDir === currentDir) {
			return null;
		}
		currentDir = parentDir;
	}
}

/**
 * Get the merge base between HEAD and a branch
 */
async function getMergeBase(
	pi: ExtensionAPI,
	branch: string,
): Promise<string | null> {
	try {
		// First try to get the upstream tracking branch
		const { stdout: upstream, code: upstreamCode } = await pi.exec("git", [
			"rev-parse",
			"--abbrev-ref",
			`${branch}@{upstream}`,
		]);

		if (upstreamCode === 0 && upstream.trim()) {
			const { stdout: mergeBase, code } = await pi.exec("git", ["merge-base", "HEAD", upstream.trim()]);
			if (code === 0 && mergeBase.trim()) {
				return mergeBase.trim();
			}
		}

		// Fall back to using the branch directly
		const { stdout: mergeBase, code } = await pi.exec("git", ["merge-base", "HEAD", branch]);
		if (code === 0 && mergeBase.trim()) {
			return mergeBase.trim();
		}

		return null;
	} catch {
		return null;
	}
}

/**
 * Get list of local branches
 */
async function getLocalBranches(pi: ExtensionAPI): Promise<string[]> {
	const { stdout, code } = await pi.exec("git", ["branch", "--format=%(refname:short)"]);
	if (code !== 0) return [];
	return stdout
		.trim()
		.split("\n")
		.filter((b) => b.trim());
}

/**
 * Parse `git log --format=%H %s` output into commits (newest first)
 */
function parseCommitLog(stdout: string): ReviewCommit[] {
	return stdout
		.trim()
		.split("\n")
		.filter((line) => line.trim())
		.map((line) => {
			const [sha, ...rest] = line.trim().split(" ");
			const title = rest.join(" ").trim();
			return title ? { sha, title } : { sha };
		});
}

/**
 * Get list of recent commits (newest first)
 */
async function getRecentCommits(pi: ExtensionAPI, limit: number = 10): Promise<ReviewCommit[]> {
	const { stdout, code } = await pi.exec("git", ["log", "--format=%H %s", "-n", `${limit}`]);
	if (code !== 0) return [];
	return parseCommitLog(stdout);
}

/**
 * Get commits newer than `sha` (exclusive, newest first)
 */
async function getCommitsAfter(
	pi: ExtensionAPI,
	sha: string,
): Promise<{ commits: ReviewCommit[]; error?: string }> {
	const { stdout, stderr, code } = await pi.exec("git", ["log", "--format=%H %s", "--end-of-options", `${sha}..HEAD`]);
	if (code !== 0) {
		return { commits: [], error: stderr.trim() || stdout.trim() || `git log failed with exit code ${code}` };
	}
	return { commits: parseCommitLog(stdout) };
}

/**
 * Resolve explicit commit shas into log order (newest first).
 * `git log --no-walk` validates the revisions; the real order comes from a
 * topology-aware walk, because `--no-walk=sorted` only sorts by committer date.
 */
async function resolveCommitList(
	pi: ExtensionAPI,
	shas: string[],
): Promise<{ commits: ReviewCommit[]; error?: string }> {
	// Fetch titles and reject revisions that do not exist.
	const picked = await pi.exec("git", [
		"log",
		"--no-walk=sorted",
		"--format=%H %s",
		"--end-of-options",
		...shas,
	]);
	if (picked.code !== 0) {
		return {
			commits: [],
			error: picked.stderr.trim() || picked.stdout.trim() || `git log failed with exit code ${picked.code}`,
		};
	}

	const selected = parseCommitLog(picked.stdout);
	if (selected.length === 0) return { commits: [] };
	if (selected.length === 1) return { commits: selected };

	// `--no-walk=sorted` orders by committer date only, which breaks for commits
	// created in the same second (or with skewed dates). Walk from the selected
	// commits and keep their first appearance, i.e. topology-aware log order.
	const tips = selected.map((commit) => commit.sha);
	const walk = await pi.exec("git", ["log", "--topo-order", "--format=%H", "--end-of-options", ...tips]);
	if (walk.code !== 0) {
		return {
			commits: [],
			error: walk.stderr.trim() || walk.stdout.trim() || `git log failed with exit code ${walk.code}`,
		};
	}

	const bySha = new Map(selected.map((commit) => [commit.sha, commit]));
	const ordered: ReviewCommit[] = [];
	const seen = new Set<string>();
	for (const line of walk.stdout.split("\n")) {
		const sha = line.trim();
		if (!bySha.has(sha) || seen.has(sha)) continue;
		seen.add(sha);
		ordered.push(bySha.get(sha)!);
	}

	return { commits: ordered };
}

/**
 * Format commits (newest first) as a prompt bullet list, oldest first.
 * `truncate` keeps the prompt small for long ranges, which is only safe when the
 * prompt also carries the exact diff command; the per-commit path needs every sha.
 */
function formatCommitList(commits: ReviewCommit[], truncate: boolean): string {
	const line = (commit: ReviewCommit) =>
		`- ${commit.sha}${commit.title ? ` ${commit.title}` : ""}`;
	const oldestFirst = [...commits].reverse();
	if (!truncate || oldestFirst.length <= 40) return oldestFirst.map(line).join("\n");

	const omitted = oldestFirst.length - 30;
	const head = oldestFirst.slice(0, 20).map(line);
	const tail = oldestFirst.slice(-10).map(line);
	return [...head, `- … (${omitted} commits omitted)`, ...tail].join("\n");
}

/**
 * Check if there are uncommitted changes (staged, unstaged, or untracked)
 */
async function hasUncommittedChanges(pi: ExtensionAPI): Promise<boolean> {
	const { stdout, code } = await pi.exec("git", ["status", "--porcelain"]);
	return code === 0 && stdout.trim().length > 0;
}

/**
 * Check if there are changes that would prevent switching branches
 * (staged or unstaged changes to tracked files - untracked files are fine)
 */
async function hasPendingChanges(pi: ExtensionAPI): Promise<boolean> {
	// Check for staged or unstaged changes to tracked files
	const { stdout, code } = await pi.exec("git", ["status", "--porcelain"]);
	if (code !== 0) return false;

	// Filter out untracked files (lines starting with ??)
	const lines = stdout.trim().split("\n").filter((line) => line.trim());
	const trackedChanges = lines.filter((line) => !line.startsWith("??"));
	return trackedChanges.length > 0;
}

/**
 * Parse a PR reference (URL or number) and return the PR number
 */
function parsePrReference(ref: string): number | null {
	const trimmed = ref.trim();

	// Try as a number first
	const num = parseInt(trimmed, 10);
	if (!isNaN(num) && num > 0) {
		return num;
	}

	// Try to extract from GitHub URL
	// Formats: https://github.com/owner/repo/pull/123
	//          github.com/owner/repo/pull/123
	const urlMatch = trimmed.match(/github\.com\/[^/]+\/[^/]+\/pull\/(\d+)/);
	if (urlMatch) {
		return parseInt(urlMatch[1], 10);
	}

	return null;
}

/**
 * Get PR information from GitHub CLI
 */
async function getPrInfo(pi: ExtensionAPI, prNumber: number): Promise<{ baseBranch: string; title: string; headBranch: string } | null> {
	const { stdout, code } = await pi.exec("gh", [
		"pr", "view", String(prNumber),
		"--json", "baseRefName,title,headRefName",
	]);

	if (code !== 0) return null;

	try {
		const data = JSON.parse(stdout);
		return {
			baseBranch: data.baseRefName,
			title: data.title,
			headBranch: data.headRefName,
		};
	} catch {
		return null;
	}
}

/**
 * Checkout a PR using GitHub CLI
 */
async function checkoutPr(pi: ExtensionAPI, prNumber: number): Promise<{ success: boolean; error?: string }> {
	const { stdout, stderr, code } = await pi.exec("gh", ["pr", "checkout", String(prNumber)]);

	if (code !== 0) {
		return { success: false, error: stderr || stdout || "Failed to checkout PR" };
	}

	return { success: true };
}

/**
 * Get the current branch name
 */
async function getCurrentBranch(pi: ExtensionAPI): Promise<string | null> {
	const { stdout, code } = await pi.exec("git", ["branch", "--show-current"]);
	if (code === 0 && stdout.trim()) {
		return stdout.trim();
	}
	return null;
}

/**
 * Get the default branch (main or master)
 */
async function getDefaultBranch(pi: ExtensionAPI): Promise<string> {
	// Try to get from remote HEAD
	const { stdout, code } = await pi.exec("git", ["symbolic-ref", "refs/remotes/origin/HEAD", "--short"]);
	if (code === 0 && stdout.trim()) {
		return stdout.trim().replace("origin/", "");
	}

	// Fall back to checking if main or master exists
	const branches = await getLocalBranches(pi);
	if (branches.includes("main")) return "main";
	if (branches.includes("master")) return "master";

	return "main"; // Default fallback
}

/**
 * Build the review prompt for one or more commits (newest first)
 */
async function buildCommitsPrompt(
	pi: ExtensionAPI,
	commits: ReviewCommit[],
	rangeBase?: string,
): Promise<string> {
	if (commits.length === 1) {
		const [commit] = commits;
		return commit.title
			? COMMIT_PROMPT_WITH_TITLE.replace("{sha}", commit.sha).replace("{title}", commit.title)
			: COMMIT_PROMPT.replace("{sha}", commit.sha);
	}

	const newest = commits[0]!;
	const oldest = commits[commits.length - 1]!;

	// A single `git diff oldest^...newest` covers the selection only when the
	// selected commits are exactly that range. Comparing counts is not enough:
	// commits outside `oldest^..newest` can make the counts coincide, which would
	// silently review unselected commits and drop selected ones.
	const { stdout, code } = await pi.exec("git", ["rev-list", `${oldest.sha}^..${newest.sha}`]);
	const inRange = new Set(
		stdout
			.split("\n")
			.map((sha) => sha.trim())
			.filter(Boolean),
	);
	const contiguous =
		code === 0 && inRange.size === commits.length && commits.every((commit) => inRange.has(commit.sha));

	// An explicit base ("every commit after X") must be used verbatim: the oldest
	// selected commit's parent can differ from X when history is non-linear.
	const prompt = rangeBase ? COMMITS_RANGE_PROMPT : contiguous ? COMMITS_DIFF_PROMPT : COMMITS_PROMPT;

	// Truncating is only safe when the prompt carries one exact diff command for
	// the whole selection; the per-commit prompt needs every sha.
	const truncate = Boolean(rangeBase) || contiguous;

	return prompt
		.replace("{count}", String(commits.length))
		.replace("{commits}", formatCommitList(commits, truncate))
		.replace("{base}", rangeBase ?? `${oldest.sha}^`)
		.replace("{newest}", newest.sha);
}

/**
 * Build the review prompt based on target
 */
async function buildReviewPrompt(
	pi: ExtensionAPI,
	target: ReviewTarget,
): Promise<string> {
	switch (target.type) {
		case "uncommitted":
			return UNCOMMITTED_PROMPT;

		case "baseBranch": {
			const mergeBase = await getMergeBase(pi, target.branch);
			const basePrompt = mergeBase
				? BASE_BRANCH_PROMPT_WITH_MERGE_BASE.replace(/{baseBranch}/g, target.branch).replace(/{mergeBaseSha}/g, mergeBase)
				: BASE_BRANCH_PROMPT_FALLBACK.replace(/{branch}/g, target.branch);
			return basePrompt;
		}

		case "commits":
			return await buildCommitsPrompt(pi, target.commits, target.rangeBase);

		case "pullRequest": {
			const mergeBase = await getMergeBase(pi, target.baseBranch);
			const basePrompt = mergeBase
				? PULL_REQUEST_PROMPT
						.replace(/{prNumber}/g, String(target.prNumber))
						.replace(/{title}/g, target.title)
						.replace(/{baseBranch}/g, target.baseBranch)
						.replace(/{mergeBaseSha}/g, mergeBase)
				: PULL_REQUEST_PROMPT_FALLBACK
						.replace(/{prNumber}/g, String(target.prNumber))
						.replace(/{title}/g, target.title)
						.replace(/{baseBranch}/g, target.baseBranch);
			return basePrompt;
		}

		case "folder":
			return FOLDER_REVIEW_PROMPT.replace("{paths}", target.paths.join(", "));
	}
}

/**
 * Get user-facing hint for the review target
 */
function getUserFacingHint(target: ReviewTarget): string {
	switch (target.type) {
		case "uncommitted":
			return "current changes";
		case "baseBranch":
			return `changes against '${target.branch}'`;
		case "commits": {
			if (target.commits.length === 1) {
				const [commit] = target.commits;
				const shortSha = commit.sha.slice(0, 7);
				return commit.title ? `commit ${shortSha}: ${commit.title}` : `commit ${shortSha}`;
			}

			const newest = target.commits[0]!.sha.slice(0, 7);
			const oldest = target.commits[target.commits.length - 1]!.sha.slice(0, 7);
			return `${target.commits.length} commits ${oldest}..${newest}`;
		}

		case "pullRequest": {
			const shortTitle = target.title.length > 30 ? target.title.slice(0, 27) + "..." : target.title;
			return `PR #${target.prNumber}: ${shortTitle}`;
		}

		case "folder": {
			const joined = target.paths.join(", ");
			return joined.length > 40 ? `folders: ${joined.slice(0, 37)}...` : `folders: ${joined}`;
		}
	}
}

// Review preset options for the selector (keep this order stable)
const REVIEW_PRESETS = [
	{ value: "uncommitted", label: "Review uncommitted changes", description: "" },
	{ value: "baseBranch", label: "Review against a base branch", description: "(local)" },
	{ value: "commit", label: "Review commits", description: "(multi-select)" },
	{ value: "pullRequest", label: "Review a pull request", description: "(GitHub PR)" },
	{ value: "folder", label: "Review a folder (or more)", description: "(snapshot, not diff)" },
] as const;

const TOGGLE_CUSTOM_INSTRUCTIONS_VALUE = "toggleCustomInstructions" as const;
type ReviewPresetValue =
	| (typeof REVIEW_PRESETS)[number]["value"]
	| typeof TOGGLE_CUSTOM_INSTRUCTIONS_VALUE;

export default function reviewExtension(pi: ExtensionAPI) {
	function persistReviewSettings() {
		pi.appendEntry(REVIEW_SETTINGS_TYPE, {
			customInstructions: reviewCustomInstructions,
		});
	}

	function setReviewCustomInstructions(instructions: string | undefined) {
		reviewCustomInstructions = instructions?.trim() || undefined;
		persistReviewSettings();
	}

	function persistReviewState(state: ReviewSessionState) {
		pi.appendEntry(REVIEW_STATE_TYPE, { ...state, branchId: state.branchId ?? randomUUID() });
	}

	// Navigations this extension starts (into/out of the review branch) restore the
	// main settings on their explicit path; the session_tree hook must not
	// duplicate that work while they run.
	let reviewNavigationDepth = 0;

	async function withReviewNavigation<T>(fn: () => Promise<T>): Promise<T> {
		reviewNavigationDepth += 1;
		try {
			return await fn();
		} finally {
			reviewNavigationDepth -= 1;
		}
	}

	async function applyAllReviewState(ctx: ExtensionContext, event?: SessionTreeEvent): Promise<void> {
		applyReviewSettings(ctx);
		const previousOriginId = reviewOriginId;
		// The branch we came from, before `applyReviewState` switches to the new one.
		// Reading it from `oldLeafId` avoids stale in-memory identity after appends.
		const previousBranch =
			event?.oldLeafId === null || event?.oldLeafId === undefined
				? undefined
				: getReviewBranchState(ctx, event.oldLeafId);
		applyReviewState(ctx);

		// Extension-started navigations (review entry/exit, auto loop) apply or
		// restore the settings on their explicit path.
		if (reviewNavigationDepth > 0) return;

		if (reviewOriginId !== undefined) {
			// A manual /tree entered a review branch (or jumped between two review
			// branches that may share the same origin). Compare the branch ids written
			// when each review started, not `originId`, which only names the return
			// target (nested reviews can even inherit ancestor state entries).
			if (previousBranch?.identity !== reviewBranchId) {
				const state = getReviewState(ctx);
				if (state?.active) {
					await restoreReviewMainSettings(ctx, state);
					await applyReviewModel(ctx, state);
				}
			}
			return;
		}

		if (previousOriginId === undefined) return;

		// A manual navigation (e.g. /tree) left the review branch without going through
		// /end-review. Read the state from the branch that was just left instead of the
		// session-wide latest entry: other review branches may have their own, newer
		// snapshot.
		const state =
			event?.oldLeafId === null || event?.oldLeafId === undefined
				? getLatestReviewSession(ctx)
				: previousBranch?.latestState;
		if (!state?.active) return;
		// Await the restore: `navigateTree` awaits session_tree handlers, so the next
		// turn (or a new review) cannot start on the review model.
		await restoreReviewMainSettings(ctx, state);
	}

	async function ensureGithubCliReady(ctx: ExtensionContext): Promise<boolean> {
		const ghVersion = await pi.exec("gh", ["--version"]);
		if (ghVersion.code !== 0) {
			ctx.ui.notify(`PR review requires GitHub CLI (\`gh\`). ${GH_SETUP_INSTRUCTIONS}`, "error");
			return false;
		}

		const ghAuthStatus = await pi.exec("gh", ["auth", "status"]);
		if (ghAuthStatus.code !== 0) {
			ctx.ui.notify(
				"GitHub CLI is installed, but you're not signed in. Run `gh auth login`, then verify with `gh auth status`.",
				"error",
			);
			return false;
		}

		return true;
	}

	async function resolvePullRequestTarget(
		ctx: ExtensionContext,
		ref: string,
		options: { skipInitialPendingChangesCheck?: boolean } = {},
	): Promise<ReviewTarget | null> {
		if (!(await ensureGithubCliReady(ctx))) {
			return null;
		}

		if (!options.skipInitialPendingChangesCheck && (await hasPendingChanges(pi))) {
			ctx.ui.notify(PR_CHECKOUT_BLOCKED_BY_PENDING_CHANGES_MESSAGE, "error");
			return null;
		}

		const prNumber = parsePrReference(ref);
		if (!prNumber) {
			ctx.ui.notify("Invalid PR reference. Enter a number or GitHub PR URL.", "error");
			return null;
		}

		ctx.ui.notify(`Fetching PR #${prNumber} info...`, "info");
		const prInfo = await getPrInfo(pi, prNumber);

		if (!prInfo) {
			ctx.ui.notify(
				`Could not fetch PR #${prNumber}. Make sure it exists and your GitHub auth has access (check with \`gh auth status\`).`,
				"error",
			);
			return null;
		}

		// Re-check right before checkout to avoid switching branches with newly introduced changes.
		if (await hasPendingChanges(pi)) {
			ctx.ui.notify(PR_CHECKOUT_BLOCKED_BY_PENDING_CHANGES_MESSAGE, "error");
			return null;
		}

		ctx.ui.notify(`Checking out PR #${prNumber}...`, "info");
		const checkoutResult = await checkoutPr(pi, prNumber);

		if (!checkoutResult.success) {
			ctx.ui.notify(`Failed to checkout PR: ${checkoutResult.error}`, "error");
			return null;
		}

		ctx.ui.notify(`Checked out PR #${prNumber} (${prInfo.headBranch})`, "info");

		return {
			type: "pullRequest",
			prNumber,
			baseBranch: prInfo.baseBranch,
			title: prInfo.title,
		};
	}

	pi.on("session_start", async (_event, ctx) => {
		await applyAllReviewState(ctx);
		const latest = readLatestAutoState(ctx);
		if (latest && latest.phase !== "done" && latest.phase !== "stopped" && latest.phase !== "error") {
			ctx.ui.notify(
				`Auto review was interrupted during ${latest.phase} (round ${latest.round}/${latest.maxRounds}). Inspect the tree and continue manually, or start a new review.`,
				"warning",
			);
		}
		setAutoWidget(ctx, undefined);
	});

	pi.on("agent_start", () => {
		autoRunStarted = true;
	});

	pi.on("agent_end", (event) => {
		autoRunSettled = true;
		for (let i = event.messages.length - 1; i >= 0; i--) {
			const message = event.messages[i];
			if (message.role !== "assistant") continue;
			if (message.stopReason === "aborted") autoRunAborted = true;
			break;
		}
	});

	pi.on("input", (event, ctx) => {
		if (!autoLoopRunning || event.source !== "interactive") return;
		ctx.ui.notify("Auto review is running. Use /review-auto-stop or Esc to stop it.", "info");
		return { action: "handled" };
	});


	pi.on("session_tree", async (event, ctx) => {
		await applyAllReviewState(ctx, event);
	});

	pi.on("session_shutdown", async (event, ctx) => {
		// pi.setModel persists the model as the global default. Quitting straight from
		// a review branch would leave the review model as the default for the next
		// session, so restore the main settings on the way out. A reload keeps this
		// session (and its review branch) alive, so leave it untouched.
		if (event.reason === "reload" || reviewNavigationDepth > 0) return;
		const state = getReviewState(ctx);
		if (!state?.active) return;
		try {
			await restoreReviewMainSettings(ctx, state);
		} catch {
			// Shutdown must not fail because the settings could not be restored.
		}
	});

	/**
	 * Determine the smart default review type based on git state
	 */
	async function getSmartDefault(): Promise<"uncommitted" | "baseBranch" | "commit"> {
		// Priority 1: If there are uncommitted changes, default to reviewing them
		if (await hasUncommittedChanges(pi)) {
			return "uncommitted";
		}

		// Priority 2: If on a feature branch (not the default branch), default to PR-style review
		const currentBranch = await getCurrentBranch(pi);
		const defaultBranch = await getDefaultBranch(pi);
		if (currentBranch && currentBranch !== defaultBranch) {
			return "baseBranch";
		}

		// Priority 3: Default to reviewing a specific commit
		return "commit";
	}

	/**
	 * Show the review preset selector
	 */
	async function showReviewSelector(ctx: ExtensionContext): Promise<ReviewTarget | null> {
		// Determine smart default (but keep the list order stable)
		const smartDefault = await getSmartDefault();
		const presetItems: SelectItem[] = REVIEW_PRESETS.map((preset) => ({
			value: preset.value,
			label: preset.label,
			description: preset.description,
		}));
		const smartDefaultIndex = presetItems.findIndex((item) => item.value === smartDefault);

		while (true) {
			const customInstructionsLabel = reviewCustomInstructions
				? "Remove custom review instructions"
				: "Add custom review instructions";
			const customInstructionsDescription = reviewCustomInstructions
				? "(currently set)"
				: "(applies to all review modes)";
			const items: SelectItem[] = [
				...presetItems,
				{
					value: TOGGLE_CUSTOM_INSTRUCTIONS_VALUE,
					label: customInstructionsLabel,
					description: customInstructionsDescription,
				},
			];

			const result = await ctx.ui.custom<ReviewPresetValue | null>((tui, theme, _kb, done) => {
				const container = new Container();
				container.addChild(new DynamicBorder((str) => theme.fg("accent", str)));
				container.addChild(new Text(theme.fg("accent", theme.bold("Select a review preset"))));

				const selectList = new SelectList(items, Math.min(items.length, 10), {
					selectedPrefix: (text) => theme.fg("accent", text),
					selectedText: (text) => theme.fg("accent", text),
					description: (text) => theme.fg("muted", text),
					scrollInfo: (text) => theme.fg("dim", text),
					noMatch: (text) => theme.fg("warning", text),
				});

				// Preselect the smart default without reordering the list
				if (smartDefaultIndex >= 0) {
					selectList.setSelectedIndex(smartDefaultIndex);
				}

				selectList.onSelect = (item) => done(item.value as ReviewPresetValue);
				selectList.onCancel = () => done(null);

				container.addChild(selectList);
				container.addChild(new Text(theme.fg("dim", "Press enter to confirm or esc to go back")));
				container.addChild(new DynamicBorder((str) => theme.fg("accent", str)));

				return {
					render(width: number) {
						return container.render(width);
					},
					invalidate() {
						container.invalidate();
					},
					handleInput(data: string) {
						selectList.handleInput(data);
						tui.requestRender();
					},
				};
			});

			if (!result) return null;

			if (result === TOGGLE_CUSTOM_INSTRUCTIONS_VALUE) {
				if (reviewCustomInstructions) {
					setReviewCustomInstructions(undefined);
					ctx.ui.notify("Custom review instructions removed", "info");
					continue;
				}

				const customInstructions = await ctx.ui.editor(
					"Enter custom review instructions (applies to all review modes):",
					"",
				);

				if (!customInstructions?.trim()) {
					ctx.ui.notify("Custom review instructions not changed", "info");
					continue;
				}

				setReviewCustomInstructions(customInstructions);
				ctx.ui.notify("Custom review instructions saved", "info");
				continue;
			}

			// Handle each preset type
			switch (result) {
				case "uncommitted":
					return { type: "uncommitted" };

				case "baseBranch": {
					const target = await showBranchSelector(ctx);
					if (target) return target;
					break;
				}

				case "commit": {
					const target = await showCommitSelector(ctx);
					if (target) return target;
					break;
				}

				case "folder": {
					const target = await showFolderInput(ctx);
					if (target) return target;
					break;
				}

				case "pullRequest": {
					const target = await showPrInput(ctx);
					if (target) return target;
					break;
				}

				default:
					return null;
			}
		}
	}

	/**
	 * Show branch selector for base branch review
	 */
	async function showBranchSelector(ctx: ExtensionContext): Promise<ReviewTarget | null> {
		const branches = await getLocalBranches(pi);
		const currentBranch = await getCurrentBranch(pi);
		const defaultBranch = await getDefaultBranch(pi);

		// Never offer the current branch as a base branch (reviewing against itself is meaningless).
		const candidateBranches = currentBranch ? branches.filter((b) => b !== currentBranch) : branches;

		if (candidateBranches.length === 0) {
			ctx.ui.notify(
				currentBranch ? `No other branches found (current branch: ${currentBranch})` : "No branches found",
				"error",
			);
			return null;
		}

		// Sort branches with default branch first
		const sortedBranches = candidateBranches.sort((a, b) => {
			if (a === defaultBranch) return -1;
			if (b === defaultBranch) return 1;
			return a.localeCompare(b);
		});

		const items: SelectItem[] = sortedBranches.map((branch) => ({
			value: branch,
			label: branch,
			description: branch === defaultBranch ? "(default)" : "",
		}));

		const result = await ctx.ui.custom<string | null>((tui, theme, keybindings, done) => {
			const container = new Container();
			container.addChild(new DynamicBorder((str) => theme.fg("accent", str)));
			container.addChild(new Text(theme.fg("accent", theme.bold("Select base branch"))));

			const searchInput = new Input();
			container.addChild(searchInput);
			container.addChild(new Spacer(1));

			const listContainer = new Container();
			container.addChild(listContainer);
			container.addChild(new Text(theme.fg("dim", "Type to filter • enter to select • esc to cancel")));
			container.addChild(new DynamicBorder((str) => theme.fg("accent", str)));

			let filteredItems = items;
			let selectList: SelectList | null = null;

			const updateList = () => {
				listContainer.clear();
				if (filteredItems.length === 0) {
					listContainer.addChild(new Text(theme.fg("warning", "  No matching branches")));
					selectList = null;
					return;
				}

				selectList = new SelectList(filteredItems, Math.min(filteredItems.length, 10), {
					selectedPrefix: (text) => theme.fg("accent", text),
					selectedText: (text) => theme.fg("accent", text),
					description: (text) => theme.fg("muted", text),
					scrollInfo: (text) => theme.fg("dim", text),
					noMatch: (text) => theme.fg("warning", text),
				});

				selectList.onSelect = (item) => done(item.value);
				selectList.onCancel = () => done(null);
				listContainer.addChild(selectList);
			};

			const applyFilter = () => {
				const query = searchInput.getValue();
				filteredItems = query
					? fuzzyFilter(items, query, (item) => `${item.label} ${item.value} ${item.description ?? ""}`)
					: items;
				updateList();
			};

			applyFilter();

			return {
				render(width: number) {
					return container.render(width);
				},
				invalidate() {
					container.invalidate();
				},
				handleInput(data: string) {
					if (
						keybindings.matches(data, "tui.select.up") ||
						keybindings.matches(data, "tui.select.down") ||
						keybindings.matches(data, "tui.select.confirm") ||
						keybindings.matches(data, "tui.select.cancel")
					) {
						if (selectList) {
							selectList.handleInput(data);
						} else if (keybindings.matches(data, "tui.select.cancel")) {
							done(null);
						}
						tui.requestRender();
						return;
					}

					searchInput.handleInput(data);
					applyFilter();
					tui.requestRender();
				},
			};
		});

		if (!result) return null;
		return { type: "baseBranch", branch: result };
	}

	/**
	 * Show commit selector (multi-select with range shortcuts)
	 */
	async function showCommitSelector(ctx: ExtensionContext): Promise<ReviewTarget | null> {
		const commits = await getRecentCommits(pi, 50);

		if (commits.length === 0) {
			ctx.ui.notify("No commits found", "error");
			return null;
		}

		const selected = new Set<string>();
		type CommitSelection = { commits: ReviewCommit[]; rangeBase?: string };

		const result = await ctx.ui.custom<CommitSelection | null>((tui, theme, keybindings, done) => {
			const container = new Container();
			container.addChild(new DynamicBorder((str) => theme.fg("accent", str)));
			container.addChild(new Text(theme.fg("accent", theme.bold("Select commit(s) to review"))));

			const searchInput = new Input();
			container.addChild(searchInput);
			container.addChild(new Spacer(1));

			const listContainer = new Container();
			container.addChild(listContainer);
			const hintText = new Text("");
			container.addChild(hintText);
			container.addChild(new DynamicBorder((str) => theme.fg("accent", str)));

			let filteredCommits = commits;
			let selectList: SelectList | null = null;

			const selectedCommits = () => commits.filter((commit) => selected.has(commit.sha));

			const updateHint = () => {
				const count = selected.size;
				const confirmHint =
					count > 0
						? `enter reviews ${count} selected commit${count === 1 ? "" : "s"}`
						: "enter reviews the highlighted commit";
				hintText.setText(
					theme.fg("dim", `space toggle • ctrl+r highlighted…newest • ${confirmHint} • esc cancel`),
				);
			};

			const buildItems = (): SelectItem[] =>
				filteredCommits.map((commit) => ({
					value: commit.sha,
					label: `${selected.has(commit.sha) ? "◉" : "○"} ${commit.sha.slice(0, 7)} ${commit.title ?? ""}`,
					description: "",
				}));

			const currentIndex = () => {
				const item = selectList?.getSelectedItem();
				if (!item) return 0;
				return Math.max(0, filteredCommits.findIndex((commit) => commit.sha === item.value));
			};

			const updateList = (index: number) => {
				listContainer.clear();
				if (filteredCommits.length === 0) {
					listContainer.addChild(new Text(theme.fg("warning", "  No matching commits")));
					selectList = null;
					updateHint();
					return;
				}

				selectList = new SelectList(buildItems(), Math.min(filteredCommits.length, 10), {
					selectedPrefix: (text) => theme.fg("accent", text),
					selectedText: (text) => theme.fg("accent", text),
					description: (text) => theme.fg("muted", text),
					scrollInfo: (text) => theme.fg("dim", text),
					noMatch: (text) => theme.fg("warning", text),
				});
				selectList.setSelectedIndex(index);

				selectList.onSelect = (item) => {
					const commit = commits.find((c) => c.sha === item.value);
					if (!commit) {
						done(null);
						return;
					}
					// Enter with a selection reviews the selection; otherwise the highlighted commit.
					done({ commits: selected.size > 0 ? selectedCommits() : [commit] });
				};
				selectList.onCancel = () => done(null);
				listContainer.addChild(selectList);
				updateHint();
			};

			const applyFilter = () => {
				const query = searchInput.getValue();
				filteredCommits = query
					? fuzzyFilter(commits, query, (commit) => `${commit.sha} ${commit.title ?? ""}`)
					: commits;
				updateList(0);
			};

			applyFilter();

			return {
				render(width: number) {
					return container.render(width);
				},
				invalidate() {
					container.invalidate();
				},
				handleInput(data: string) {
					if (matchesKey(data, Key.space)) {
						const item = selectList?.getSelectedItem();
						if (item) {
							const index = currentIndex();
							if (selected.has(item.value)) {
								selected.delete(item.value);
							} else {
								selected.add(item.value);
							}
							updateList(index);
							tui.requestRender();
						}
						return;
					}

					// "Review commits after this one": the highlighted commit is the
					// exclusive base, so everything newer in the log is included.
					if (matchesKey(data, Key.ctrl("r"))) {
						const item = selectList?.getSelectedItem();
						if (item) {
							const index = commits.findIndex((commit) => commit.sha === item.value);
							const newer = index > 0 ? commits.slice(0, index) : [];
							if (newer.length === 0) {
								ctx.ui.notify("No commits newer than the highlighted commit", "info");
							} else {
								done({ commits: newer, rangeBase: item.value });
							}
						}
						return;
					}

					if (
						keybindings.matches(data, "tui.select.up") ||
						keybindings.matches(data, "tui.select.down") ||
						keybindings.matches(data, "tui.select.confirm") ||
						keybindings.matches(data, "tui.select.cancel")
					) {
						if (selectList) {
							selectList.handleInput(data);
						} else if (keybindings.matches(data, "tui.select.cancel")) {
							done(null);
						}
						tui.requestRender();
						return;
					}

					searchInput.handleInput(data);
					applyFilter();
					tui.requestRender();
				},
			};
		});

		if (!result) return null;
		// `getRecentCommits` returns plain git log order, so the newest/oldest
		// boundaries used by the prompt and hint are re-resolved topology-aware.
		const { commits: ordered, error } = await resolveCommitList(
			pi,
			result.commits.map((commit) => commit.sha),
		);
		if (error) {
			ctx.ui.notify(`Failed to resolve commits: ${error}`, "error");
			return null;
		}
		if (ordered.length === 0) {
			ctx.ui.notify("No commits found for the selection", "error");
			return null;
		}
		return { type: "commits", commits: ordered, rangeBase: result.rangeBase };
	}


	function parseReviewPaths(value: string): string[] {
		return value
			.split(/\s+/)
			.map((item) => item.trim())
			.filter((item) => item.length > 0);
	}

	/**
	 * Show folder input
	 */
	async function showFolderInput(ctx: ExtensionContext): Promise<ReviewTarget | null> {
		const result = await ctx.ui.editor(
			"Enter folders/files to review (space-separated or one per line):",
			".",
		);

		if (!result?.trim()) return null;
		const paths = parseReviewPaths(result);
		if (paths.length === 0) return null;

		return { type: "folder", paths };
	}

	/**
	 * Show PR input and handle checkout
	 */
	async function showPrInput(ctx: ExtensionContext): Promise<ReviewTarget | null> {
		// First check for pending changes that would prevent branch switching
		if (await hasPendingChanges(pi)) {
			ctx.ui.notify(PR_CHECKOUT_BLOCKED_BY_PENDING_CHANGES_MESSAGE, "error");
			return null;
		}

		// Get PR reference from user
		const prRef = await ctx.ui.editor(
			"Enter PR number or URL (e.g. 123 or https://github.com/owner/repo/pull/123):",
			"",
		);

		if (!prRef?.trim()) return null;

		return await resolvePullRequestTarget(ctx, prRef, { skipInitialPendingChangesCheck: true });
	}

	/**
	 * Execute the review
	 */
	async function enterFreshReviewBranch(ctx: ExtensionCommandContext): Promise<string | null> {
		// Store current position (where we'll return to).
		// In an empty session there is no leaf yet, so create a lightweight anchor first.
		let originId = ctx.sessionManager.getLeafId() ?? undefined;
		if (!originId) {
			pi.appendEntry(REVIEW_ANCHOR_TYPE, { createdAt: new Date().toISOString() });
			originId = ctx.sessionManager.getLeafId() ?? undefined;
		}
		if (!originId) {
			ctx.ui.notify("Failed to determine review origin.", "error");
			return null;
		}
		reviewOriginId = originId;

		// Keep a local copy so session_tree events during navigation don't wipe it
		const lockedOriginId = originId;

		// Find the first user message in the session.
		// If none exists (e.g. brand-new session), we'll stay on the current leaf.
		const entries = ctx.sessionManager.getEntries();
		const firstUserMessage = entries.find(
			(e) => e.type === "message" && e.message.role === "user",
		);

		if (firstUserMessage) {
			// Navigate to first user message to create a new branch from that point
			// Label it as "code-review" so it's visible in the tree
			try {
				const result = await withReviewNavigation(() =>
					ctx.navigateTree(firstUserMessage.id, { summarize: false, label: "code-review" }),
				);
				if (result.cancelled) {
					reviewOriginId = undefined;
					return null;
				}
			} catch (error) {
				// Clean up state if navigation fails
				reviewOriginId = undefined;
				ctx.ui.notify(`Failed to start review: ${error instanceof Error ? error.message : String(error)}`, "error");
				return null;
			}

			// Clear the editor (navigating to user message fills it with the message text)
			ctx.ui.setEditorText("");
		}

		// Restore origin after navigation events (session_tree can reset it)
		reviewOriginId = lockedOriginId;

		// Show widget indicating review is active
		setReviewWidget(ctx, true);

		return lockedOriginId;
	}

	async function buildFullReviewPrompt(
		ctx: ExtensionCommandContext,
		target: ReviewTarget,
		extraInstruction?: string,
		auto?: { round: number },
	): Promise<string> {
		const prompt = await buildReviewPrompt(pi, target);
		const projectGuidelines = await loadProjectReviewGuidelines(ctx.cwd);

		// Combine the review rubric with the specific prompt
		let fullPrompt = `${REVIEW_RUBRIC}\n\n---\n\nPlease perform a code review with the following focus:\n\n${prompt}`;

		if (reviewCustomInstructions) {
			fullPrompt += `\n\nShared custom review instructions (applies to all reviews):\n\n${reviewCustomInstructions}`;
		}

		if (extraInstruction?.trim()) {
			fullPrompt += `\n\nAdditional user-provided review instruction:\n\n${extraInstruction.trim()}`;
		}

		if (projectGuidelines) {
			fullPrompt += `\n\nThis project has additional instructions for code reviews:\n\n${projectGuidelines}`;
		}

		// Keep last so it overrides any other language guidance
		fullPrompt += `\n\n${REVIEW_OUTPUT_LANGUAGE}`;

		if (auto) {
			fullPrompt += `\n\n${buildAutoProtocol(auto.round, "review", [])}`;
		}

		return fullPrompt;
	}

	async function executeReview(
		ctx: ExtensionCommandContext,
		target: ReviewTarget,
		useFreshSession: boolean,
		options?: { extraInstruction?: string; reviewModel?: ReviewModel },
	): Promise<boolean> {
		// Check if we're already in a review
		if (reviewOriginId) {
			ctx.ui.notify("Already in a review. Use /end-review to finish first.", "warning");
			return false;
		}

		// Captured before the switch so /end-review can restore the main settings.
		const mainModel = modelRef(ctx.model);
		const mainThinkingLevel = pi.getThinkingLevel();

		let reviewOriginIdForState: string | undefined;
		if (useFreshSession) {
			reviewOriginIdForState = (await enterFreshReviewBranch(ctx)) ?? undefined;
			if (!reviewOriginIdForState) return false;
		}

		let appliedReviewModel: ModelRef | undefined;
		if (options?.reviewModel) {
			if (await applyModel(options.reviewModel)) {
				appliedReviewModel = modelRef(options.reviewModel);
			} else {
				ctx.ui.notify(
					`No authentication for ${options.reviewModel.provider}/${options.reviewModel.id}; using the current model for the review branch.`,
					"warning",
				);
			}
		}

		if (reviewOriginIdForState) {
			// Persist one state entry per review branch: its id is the branch identity, and
			// it carries the review model so `/tree` re-entry can reapply it (tree
			// navigation does not replay model_change entries).
			persistReviewState({
				active: true,
				originId: reviewOriginIdForState,
				mainModel,
				mainThinkingLevel,
				reviewModel: appliedReviewModel,
			});
		}

		const fullPrompt = await buildFullReviewPrompt(ctx, target, options?.extraInstruction);
		const hint = getUserFacingHint(target);

		const modeHint = useFreshSession ? " (fresh session)" : "";
		ctx.ui.notify(`Starting review: ${hint}${modeHint}`, "info");

		// Send as a user message that triggers a turn
		pi.sendUserMessage(fullPrompt);
		return true;
	}

	// ---------------------------------------------------------------------
	// Auto review loop (review -> fix -> re-review until clean)
	// ---------------------------------------------------------------------

	const AUTO_FIX_SUMMARY_PROMPT = `We are leaving the main coding branch and returning to the code-review branch for verification.
Summarize what was changed to address the review findings. Include:
- Which findings were addressed and how
- Which findings were skipped/deferred and why
- Tests/checks that were run and their results
- Any new risks or behavior changes introduced by the fixes
Keep it concise and factual; the reviewer will verify against the actual files.`;

	function persistAutoState(snapshot: AutoSnapshot): void {
		pi.appendEntry(AUTO_STATE_TYPE, { ...snapshot, updatedAt: new Date().toISOString() });
	}

	function readLatestAutoState(ctx: ExtensionContext): AutoSnapshot | undefined {
		let latest: AutoSnapshot | undefined;
		for (const entry of ctx.sessionManager.getEntries()) {
			if (entry.type !== "custom" || entry.customType !== AUTO_STATE_TYPE) continue;
			const data = entry.data as AutoSnapshot | undefined;
			if (data && data.v === 1) latest = data;
		}
		return latest;
	}

	function setAutoWidget(ctx: ExtensionContext, snapshot: AutoSnapshot | undefined): void {
		if (!ctx.hasUI) return;
		if (!snapshot) {
			ctx.ui.setWidget("review-auto", undefined);
			return;
		}

		const phaseLabel =
			snapshot.phase === "review"
				? "reviewing"
				: snapshot.phase === "fix"
					? "fixing findings"
					: snapshot.phase;

		ctx.ui.setWidget("review-auto", (_tui, theme) => {
			const text = new Text(
				theme.fg(
					"accent",
					`Auto review · round ${snapshot.round}/${snapshot.maxRounds} · ${phaseLabel} · /review-auto-stop to stop`,
				),
				0,
				0,
			);
			return {
				render(width: number) {
					return text.render(width);
				},
				invalidate() {
					text.invalidate();
				},
			};
		});
	}

	function resolveStoredModel(ctx: ExtensionContext, ref: ModelRef | undefined): ReviewModel | undefined {
		if (!ref) return undefined;
		return ctx.modelRegistry.find(ref.provider, ref.id);
	}

	function findModelByRef(ctx: ExtensionContext, ref: string): ReviewModel | undefined {
		const slash = ref.indexOf("/");
		if (slash > 0) {
			const found = ctx.modelRegistry.find(ref.slice(0, slash), ref.slice(slash + 1));
			if (found) return found;
		}
		return ctx.modelRegistry.getAvailable().find((model) => model.id === ref || `${model.provider}/${model.id}` === ref);
	}

	function modelRef(model: ReviewModel | undefined): ModelRef | undefined {
		return model ? { provider: model.provider, id: model.id } : undefined;
	}

	// Distinguishes "nothing was saved" from "saved model can no longer be
	// resolved": the latter is a real failure and must not be reported as a
	// successful restore while the review model stays active.
	type StoredModelApply = "none" | "ok" | "missing" | "failed";

	async function applyStoredModel(
		ctx: ExtensionContext,
		ref: ModelRef | undefined,
	): Promise<StoredModelApply> {
		if (!ref) return "none";
		const model = resolveStoredModel(ctx, ref);
		if (!model) return "missing";
		return (await applyModel(model)) ? "ok" : "failed";
	}

	async function applyModel(model: ReviewModel | undefined): Promise<boolean> {
		if (!model) return true;
		try {
			return await pi.setModel(model);
		} catch {
			return false;
		}
	}

	/**
	 * Restore the main-branch model and thinking level that a review branch
	 * replaced. Shared by /end-review, manual tree navigation, and the auto loop so
	 * every exit path keeps the same ordering (model first, then thinking level)
	 * and the same failure semantics.
	 */
	async function restoreMainSettings(
		ctx: ExtensionContext,
		mainModel: ModelRef | undefined,
		mainThinkingLevel: ReviewThinkingLevel | undefined,
	): Promise<StoredModelApply> {
		let result: StoredModelApply = "none";
		if (mainModel) {
			const current = ctx.model;
			const unchanged =
				current !== undefined && current.provider === mainModel.provider && current.id === mainModel.id;
			result = unchanged ? "ok" : await applyStoredModel(ctx, mainModel);
		}

		// A failed model switch leaves the review model active; its thinking-level
		// clamp would only be changed again, so surface the model failure instead.
		if (result === "missing" || result === "failed") return result;

		if (mainThinkingLevel && pi.getThinkingLevel() !== mainThinkingLevel) {
			applyThinking(mainThinkingLevel);
		}
		return result;
	}

	/**
	 * Restore the main-branch settings recorded on a review-session state entry and
	 * report failures to the user. Used by /end-review and when a manual tree
	 * navigation leaves the review branch.
	 */
	async function restoreReviewMainSettings(
		ctx: ExtensionContext,
		state: ReviewSessionState | undefined,
	): Promise<void> {
		if (!state) return;

		const result = await restoreMainSettings(ctx, state.mainModel, state.mainThinkingLevel);
		if (result !== "missing" && result !== "failed") return;

		const label = state.mainModel
			? `${state.mainModel.provider}/${state.mainModel.id}`
			: "the main branch model";
		ctx.ui.notify(
			result === "missing"
				? `Could not restore ${label}: the model is no longer available. The review model is still active.`
				: `Could not restore ${label}; the review model is still active.`,
			"warning",
		);
	}

	/**
	 * Reapply the review-only model when a manual navigation enters a review branch.
	 * A branch without a saved review model ("use current model") is left untouched.
	 */
	async function applyReviewModel(
		ctx: ExtensionContext,
		state: ReviewSessionState | undefined,
	): Promise<void> {
		if (!state?.active) return;

		const result = await applyStoredModel(ctx, state.reviewModel);
		if (result !== "missing" && result !== "failed") return;

		const label = state.reviewModel
			? `${state.reviewModel.provider}/${state.reviewModel.id}`
			: "the review model";
		ctx.ui.notify(
			result === "missing"
				? `Could not switch to ${label}: the model is no longer available. The current model is still in use.`
				: `Could not switch to ${label}; the current model is still in use.`,
			"warning",
		);
	}

	function applyThinking(level: ReviewThinkingLevel | undefined): void {
		if (!level) return;
		// `pi.setThinkingLevel` clamps to what the current model supports and does
		// not throw. Swallowing errors here would hide real failures and silently
		// keep the wrong thinking level for the remaining rounds.
		pi.setThinkingLevel(level);
	}

	async function gitOutput(args: string[]): Promise<string> {
		const result = await pi.exec("git", args);
		if (result.code !== 0) {
			throw new Error(
				`git ${args.join(" ")} failed (exit code ${result.code}): ${result.stderr.trim()}`,
			);
		}
		return result.stdout;
	}

	async function getWorkingTreeHash(): Promise<string> {
		const hash = createHash("sha1");
		// Include HEAD so a fix agent that commits despite the instructions is still
		// visible as progress. `--verify` fails on an unborn HEAD (repository with
		// no commits yet), which is an expected state, not an error.
		const head = await pi.exec("git", ["rev-parse", "--verify", "HEAD"]);
		hash.update(head.code === 0 ? head.stdout.trim() : "unborn");
		hash.update("\u0000");
		// `git diff` covers unstaged changes and `git diff --cached` covers staged
		// changes. Together they match `git diff HEAD` but also work in a repository
		// with no commits yet (unborn HEAD).
		hash.update(await gitOutput(["diff"]));
		hash.update("\u0000");
		hash.update(await gitOutput(["diff", "--cached"]));
		hash.update("\u0000");

		const untracked = (await gitOutput(["ls-files", "--others", "--exclude-standard", "-z"]))
			.split("\u0000")
			.filter((file) => file.length > 0)
			.sort();
		if (untracked.length > 0) {
			hash.update(untracked.join("\u0000"));
			hash.update("\u0000");
			hash.update(await gitOutput(["hash-object", "--no-filters", "--", ...untracked]));
		}

		// Ignored paths are deliberately excluded from progress detection: test and
		// build runs create/delete them without the fix agent changing anything, so
		// including them would defeat the no-progress stop condition.
		return hash.digest("hex");
	}

	async function navigateIdleTo(
		ctx: ExtensionCommandContext,
		targetId: string,
		summaryPrompt?: string,
	): Promise<boolean> {
		await ctx.waitForIdle();
		if (summaryPrompt) {
			// --summary is an explicit request, so a failed or impossible summary is
			// surfaced instead of silently degrading to an unsummarized navigation.
			// Expected failure modes: no model selected, or the summarization call
			// itself failing (e.g. auth/provider error). Both propagate to the caller.
			if (!ctx.model) {
				throw new Error("cannot summarize the review branch: no model is available for the summary");
			}
			const result = await withReviewNavigation(() =>
				ctx.navigateTree(targetId, {
					summarize: true,
					customInstructions: summaryPrompt,
					replaceInstructions: true,
				}),
			);
			return !result.cancelled;
		}
		const result = await withReviewNavigation(() => ctx.navigateTree(targetId, { summarize: false }));
		return !result.cancelled;
	}

	async function sendAndWait(
		ctx: ExtensionCommandContext,
		prompt: string,
	): Promise<{ aborted: boolean; producedAssistant: boolean }> {
		autoRunStarted = false;
		autoRunSettled = false;
		autoRunAborted = false;
		const leafBefore = ctx.sessionManager.getLeafId();
		const assistantBefore = getLastAssistantEntry(ctx)?.id;
		pi.sendUserMessage(prompt);

		// `pi.sendUserMessage` is fire-and-forget, so wait until the run is
		// observable before waiting for it to settle.
		const deadline = Date.now() + 20_000;
		while (!autoRunStarted && !autoRunSettled && ctx.isIdle() && Date.now() < deadline) {
			await sleep(25);
		}

		const leafAfter = ctx.sessionManager.getLeafId();
		const runObserved =
			autoRunStarted || autoRunSettled || !ctx.isIdle() || leafAfter !== leafBefore;
		if (!runObserved) {
			throw new Error("The agent run did not start (the message may have been intercepted).");
		}
		if (!ctx.isIdle()) {
			await ctx.waitForIdle();
		}

		const assistantAfter = getLastAssistantEntry(ctx);
		return {
			aborted: autoRunAborted || autoStopRequested,
			producedAssistant: assistantAfter !== undefined && assistantAfter.id !== assistantBefore,
		};
	}

	async function showReviewModelSelector(
		ctx: ExtensionCommandContext,
	): Promise<{ model?: ReviewModel } | null> {
		const current = ctx.model;
		const currentLabel = current ? `${current.provider}/${current.id}` : "current model";
		// `scopedModels` exists on newer Pi versions; fall back to the full catalogue.
		const maybeScoped = (ctx as unknown as { scopedModels?: Array<{ model: ReviewModel }> }).scopedModels;
		const candidates = (maybeScoped && maybeScoped.length > 0 ? maybeScoped.map((entry) => entry.model) : ctx.modelRegistry.getAvailable())
			.slice()
			.sort((a, b) => `${a.provider}/${a.id}`.localeCompare(`${b.provider}/${b.id}`));

		const items = [`Use current model (${currentLabel})`];
		const seen = new Set<string>([currentLabel]);
		for (const model of candidates) {
			const key = `${model.provider}/${model.id}`;
			if (seen.has(key)) continue;
			seen.add(key);
			items.push(key);
		}

		const choice = await ctx.ui.select("Model for the review branch:", items);
		if (choice === undefined) return null;
		if (choice === items[0]) return {};

		const model = findModelByRef(ctx, choice);
		if (!model) {
			ctx.ui.notify(`Model not found: ${choice}`, "error");
			return null;
		}
		return { model };
	}

	async function showAutoMaxRoundsSelector(ctx: ExtensionCommandContext): Promise<number | null> {
		const choice = await ctx.ui.select("Max auto-review rounds:", ["3", "5", "10"]);
		if (choice === undefined) return null;
		return Number.parseInt(choice, 10);
	}

	/** Whether the current leaf still lives on the review branch. */
	function isOnReviewBranch(ctx: ExtensionContext, snapshot: AutoSnapshot): boolean {
		const reviewLeafId = snapshot.reviewLeafId;
		if (!reviewLeafId) return snapshot.branch === "review";
		return ctx.sessionManager.getBranch().some((entry) => entry.id === reviewLeafId);
	}

	async function finishAutoReview(
		ctx: ExtensionCommandContext,
		snapshot: AutoSnapshot,
		phase: AutoPhase,
		reason: string,
	): Promise<void> {
		try {
			// Trust the tree, not the snapshot: the caller may have navigated just
			// before an error, leaving snapshot.branch stale.
			if (isOnReviewBranch(ctx, snapshot)) {
				const reviewLeafId = ctx.sessionManager.getLeafId() ?? snapshot.reviewLeafId;
				persistAutoState({ ...snapshot, phase, reason, reviewLeafId });
				pi.appendEntry(REVIEW_STATE_TYPE, { active: false });
				if (!(await navigateIdleTo(ctx, snapshot.mainLeafId))) {
					ctx.ui.notify(
						"Auto review could not return to the main branch: navigation was cancelled. You are still on the review branch.",
						"warning",
					);
					persistAutoState({ ...snapshot, phase: "error", reason: `${reason} (navigation cancelled)` });
					return;
				}
			}

			const modelResult = await restoreMainSettings(ctx, snapshot.mainModel, snapshot.mainThinkingLevel);
			if (modelResult === "missing" || modelResult === "failed") {
				const label = snapshot.mainModel
					? `${snapshot.mainModel.provider}/${snapshot.mainModel.id}`
					: "the main branch model";
				ctx.ui.notify(
					modelResult === "missing"
						? `Auto review could not restore ${label}: the model is no longer available. The current model is still in use.`
						: `Auto review could not restore ${label}; the current model is still in use.`,
					"error",
				);
				// We are back on the main branch, so leaving the review state behind
				// would make the next /review report "Already in a review".
				persistAutoState({
					...snapshot,
					branch: "main",
					phase: "error",
					reason: `failed to restore main branch model (${reason})`,
				});
				clearReviewState(ctx);
				return;
			}

			const mainLeafId = ctx.sessionManager.getLeafId() ?? snapshot.mainLeafId;
			persistAutoState({ ...snapshot, branch: "main", phase, reason, mainLeafId });
			clearReviewState(ctx);
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`Failed to finish auto review cleanly: ${message}`, "warning");

			// Navigation may already have succeeded before this failure. Read the
			// actual position from the session tree instead of trusting the stale
			// snapshot, so an error on the main branch does not leave the session
			// marked "in review" (which would block the next /review).
			const onReviewBranch = isOnReviewBranch(ctx, snapshot);
			try {
				persistAutoState({
					...snapshot,
					branch: onReviewBranch ? "review" : "main",
					phase: "error",
					reason: `${reason}: ${message}`,
					mainLeafId: onReviewBranch
						? snapshot.mainLeafId
						: (ctx.sessionManager.getLeafId() ?? snapshot.mainLeafId),
				});
			} catch (persistError) {
				ctx.ui.notify(
					`Failed to record the auto review error state: ${persistError instanceof Error ? persistError.message : String(persistError)}`,
					"warning",
				);
			}

			if (!onReviewBranch) {
				try {
					clearReviewState(ctx);
				} catch (cleanupError) {
					ctx.ui.notify(
						`Failed to clear the review state: ${cleanupError instanceof Error ? cleanupError.message : String(cleanupError)}`,
						"warning",
					);
				}
			}
		} finally {
			setAutoWidget(ctx, undefined);
			setReviewWidget(ctx, false);
		}
	}

	async function runAutoReviewLoop(
		ctx: ExtensionCommandContext,
		target: ReviewTarget,
		options: AutoReviewOptions,
	): Promise<void> {
		if (autoLoopRunning) {
			ctx.ui.notify("Auto review is already running. Use /review-auto-stop to stop it.", "warning");
			return;
		}

		autoLoopRunning = true;
		autoStopRequested = false;

		const runId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
		const startedAt = new Date().toISOString();
		const mainModel = ctx.model;
		const mainThinkingLevel = pi.getThinkingLevel();
		let reviewModel = options.reviewModel;
		const hint = getUserFacingHint(target);
		let reviewModelWarned = false;
		let snapshot: AutoSnapshot | undefined;

		try {
			const originId = await enterFreshReviewBranch(ctx);
			if (!originId) return;

			if (reviewModel && !(await applyModel(reviewModel)) && !reviewModelWarned) {
				reviewModelWarned = true;
				ctx.ui.notify(
					`No authentication for ${reviewModel.provider}/${reviewModel.id}; using the current model for the review branch.`,
					"warning",
				);
				reviewModel = undefined;
			}

			// Record the review model on the branch so manual /tree re-entry can reapply it.
			// This is the single state entry that defines the review branch.
			persistReviewState({
				active: true,
				originId,
				mainModel: modelRef(mainModel),
				mainThinkingLevel,
				reviewModel: modelRef(reviewModel),
			});

			snapshot = {
				v: 1,
				runId,
				branch: "review",
				phase: "review",
				round: 1,
				maxRounds: options.maxRounds,
				target,
				originId,
				mainLeafId: originId,
				reviewLeafId: ctx.sessionManager.getLeafId() ?? originId,
				mainModel: modelRef(mainModel),
				mainThinkingLevel,
				reviewModel: modelRef(reviewModel),
				startedAt,
				updatedAt: startedAt,
			};
			persistAutoState(snapshot);
			snapshot.reviewLeafId = ctx.sessionManager.getLeafId() ?? snapshot.reviewLeafId;
			setAutoWidget(ctx, snapshot);
			ctx.ui.notify(`Auto review started: ${hint} (max ${options.maxRounds} rounds)`, "info");

			let round = 1;
			let previousKeys = "";
			let lastFixChangedTree = false;

			let run = await sendAndWait(
				ctx,
				await buildFullReviewPrompt(ctx, target, options.extraInstruction, { round: 1 }),
			);
			if (autoStopRequested || run.aborted) {
				await finishAutoReview(ctx, snapshot, "stopped", "stopped by user");
				return;
			}
			if (!run.producedAssistant) {
				ctx.ui.notify("Auto review stopped: the review run produced no assistant response.", "error");
				await finishAutoReview(ctx, snapshot, "error", "review produced no assistant response");
				return;
			}

			while (true) {
				if (autoStopRequested) {
					await finishAutoReview(ctx, snapshot, "stopped", "stopped by user");
					return;
				}

				const assistant = getLastAssistantEntry(ctx);
				if (!assistant || assistant.stopReason === "error") {
					ctx.ui.notify("Auto review stopped: the review run failed. Inspect the transcript and continue manually.", "error");
					await finishAutoReview(ctx, snapshot, "error", "review run failed");
					return;
				}

				const parsed = parseAutoReviewResult(assistant.text);
				if (!parsed.ok) {
					const description = describeAutoReviewParseFailure(parsed.reason);
					ctx.ui.notify(
						`Auto review stopped: ${description}${parsed.detail ? ` (${parsed.detail})` : ""}. Continue manually with /end-review.`,
						"warning",
					);
					await finishAutoReview(ctx, snapshot, "error", `invalid REVIEW_RESULT: ${parsed.reason}`);
					return;
				}
				const review = parsed.result;

				let reviewLeafId: string =
					ctx.sessionManager.getLeafId() ?? snapshot.reviewLeafId ?? snapshot.originId;
				snapshot = {
					...snapshot,
					branch: "review",
					phase: "review",
					round,
					reviewLeafId,
					lastVerdict: review.verdict,
					findingCount: review.findings.length,
					findingKeys: review.findings.map(findingKey),
				};
				persistAutoState(snapshot);
				reviewLeafId = ctx.sessionManager.getLeafId() ?? reviewLeafId;
				snapshot = { ...snapshot, reviewLeafId };
				setAutoWidget(ctx, snapshot);

				if (review.findings.length === 0) {
					await finishAutoReview(ctx, snapshot, "done", "no findings");
					ctx.ui.notify(`Auto review complete: no findings after ${round} round(s).`, "info");
					return;
				}

				const keys = review.findings.map(findingKey).sort().join("|");
				if (round > 1 && keys === previousKeys && !lastFixChangedTree) {
					ctx.ui.notify(
						`Auto review stopped: findings did not change and the fix round made no file changes (round ${round}).`,
						"warning",
					);
					await finishAutoReview(ctx, snapshot, "stopped", "no progress");
					return;
				}
				previousKeys = keys;

				if (round >= options.maxRounds) {
					ctx.ui.notify(
						`Auto review stopped after ${round} round(s); ${review.findings.length} finding(s) remain.`,
						"warning",
					);
					await finishAutoReview(ctx, snapshot, "stopped", `max rounds reached (${round})`);
					return;
				}

				// Return to the main branch and fix.
				if (autoStopRequested) {
					await finishAutoReview(ctx, snapshot, "stopped", "stopped by user");
					return;
				}
				if (!(await navigateIdleTo(ctx, snapshot.mainLeafId, options.summary ? REVIEW_SUMMARY_PROMPT : undefined))) {
					ctx.ui.notify("Auto review stopped: returning to the main branch was cancelled.", "warning");
					await finishAutoReview(ctx, snapshot, "stopped", "navigation to main branch cancelled");
					return;
				}
				const mainModelResult = await restoreMainSettings(
					ctx,
					snapshot.mainModel,
					snapshot.mainThinkingLevel,
				);
				if (mainModelResult === "missing" || mainModelResult === "failed") {
					// finishAutoReview retries the restore and surfaces the failure to the user.
					await finishAutoReview(ctx, { ...snapshot, branch: "main" }, "error", "failed to restore main branch model");
					return;
				}

				let mainLeafId: string = ctx.sessionManager.getLeafId() ?? snapshot.mainLeafId;
				snapshot = { ...snapshot, branch: "main", phase: "fix", mainLeafId, reviewLeafId };
				persistAutoState(snapshot);
				mainLeafId = ctx.sessionManager.getLeafId() ?? mainLeafId;
				snapshot = { ...snapshot, mainLeafId };
				setAutoWidget(ctx, snapshot);

				const hashBeforeFix = await getWorkingTreeHash();
				run = await sendAndWait(ctx, buildAutoFixPrompt(review.findings));
				if (autoStopRequested || run.aborted) {
					await finishAutoReview(ctx, snapshot, "stopped", "stopped by user");
					return;
				}
				if (!run.producedAssistant) {
					ctx.ui.notify("Auto review stopped: the fix run produced no assistant response.", "error");
					await finishAutoReview(ctx, snapshot, "error", "fix produced no assistant response");
					return;
				}

				const fixAssistant = getLastAssistantEntry(ctx);
				if (!fixAssistant || fixAssistant.stopReason === "error") {
					ctx.ui.notify("Auto review stopped: the fix run failed. Inspect the working tree and continue manually.", "error");
					await finishAutoReview(ctx, snapshot, "error", "fix run failed");
					return;
				}

				const hashAfterFix = await getWorkingTreeHash();
				lastFixChangedTree = hashBeforeFix !== hashAfterFix;
				snapshot = {
					...snapshot,
					phase: "review",
					round: round + 1,
					mainLeafId: ctx.sessionManager.getLeafId() ?? snapshot.mainLeafId,
					lastFixChangedTree,
				};
				persistAutoState(snapshot);
				snapshot.mainLeafId = ctx.sessionManager.getLeafId() ?? snapshot.mainLeafId;

				// Return to the review branch and verify.
				if (autoStopRequested) {
					await finishAutoReview(ctx, snapshot, "stopped", "stopped by user");
					return;
				}
				if (!(await navigateIdleTo(ctx, reviewLeafId, options.summary ? AUTO_FIX_SUMMARY_PROMPT : undefined))) {
					ctx.ui.notify("Auto review stopped: returning to the review branch was cancelled.", "warning");
					await finishAutoReview(ctx, snapshot, "stopped", "navigation to review branch cancelled");
					return;
				}
				if (reviewModel && !(await applyModel(reviewModel)) && !reviewModelWarned) {
					reviewModelWarned = true;
					ctx.ui.notify(
						`No authentication for ${reviewModel.provider}/${reviewModel.id}; continuing verification with the current model.`,
						"warning",
					);
					reviewModel = undefined;
				}
				if (!reviewModel) {
					applyThinking(snapshot.mainThinkingLevel);
				}

				round += 1;
				const nextReviewLeafId: string = ctx.sessionManager.getLeafId() ?? reviewLeafId;
				snapshot = { ...snapshot, branch: "review", phase: "review", round, reviewLeafId: nextReviewLeafId };
				persistAutoState(snapshot);
				reviewLeafId = ctx.sessionManager.getLeafId() ?? nextReviewLeafId;
				snapshot = { ...snapshot, reviewLeafId };
				setAutoWidget(ctx, snapshot);

				if (autoStopRequested) {
					await finishAutoReview(ctx, snapshot, "stopped", "stopped by user");
					return;
				}

				run = await sendAndWait(
					ctx,
					`${buildAutoProtocol(round, "verify", review.findings)}${
						fixAssistant.text.trim() ? `\n\nFix report from the implementing agent:\n${fixAssistant.text.trim().slice(0, 4000)}` : ""
					}\n\n${REVIEW_OUTPUT_LANGUAGE}`,
				);
				if (autoStopRequested || run.aborted) {
					await finishAutoReview(ctx, snapshot, "stopped", "stopped by user");
					return;
				}
				if (!run.producedAssistant) {
					ctx.ui.notify("Auto review stopped: the verification run produced no assistant response.", "error");
					await finishAutoReview(ctx, snapshot, "error", "verification produced no assistant response");
					return;
				}
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error);
			ctx.ui.notify(`Auto review failed: ${message}`, "error");
			if (snapshot) {
				await finishAutoReview(ctx, snapshot, "error", message);
			}
		} finally {
			autoLoopRunning = false;
			setAutoWidget(ctx, undefined);
		}
	}

	/**
	 * Parse command arguments for direct invocation
	 * Returns the target or a special marker for PR that needs async handling
	 */
	type ParsedAutoFlags = {
		reviewModel?: string;
		maxRounds?: number;
		summary: boolean;
	};

	type ParsedReviewArgs = {
		target:
			| ReviewTarget
			| { type: "pr"; ref: string }
			| { type: "commitsAfter"; sha: string }
			| { type: "commitsList"; shas: string[] }
			| null;
		extraInstruction?: string;
		auto?: ParsedAutoFlags;
		error?: string;
	};

	function tokenizeArgs(value: string): string[] {
		const tokens: string[] = [];
		let current = "";
		let quote: '"' | "'" | null = null;

		for (let i = 0; i < value.length; i++) {
			const char = value[i];

			if (quote) {
				if (char === "\\" && i + 1 < value.length) {
					current += value[i + 1];
					i += 1;
					continue;
				}
				if (char === quote) {
					quote = null;
					continue;
				}
				current += char;
				continue;
			}

			if (char === '"' || char === "'") {
				quote = char;
				continue;
			}

			if (/\s/.test(char)) {
				if (current.length > 0) {
					tokens.push(current);
					current = "";
				}
				continue;
			}

			current += char;
		}

		if (current.length > 0) {
			tokens.push(current);
		}

		return tokens;
	}

	function parseArgs(args: string | undefined): ParsedReviewArgs {
		if (!args?.trim()) return { target: null };

		const rawParts = tokenizeArgs(args.trim());
		const parts: string[] = [];
		let extraInstruction: string | undefined;
		let autoRequested = false;
		let autoReviewModel: string | undefined;
		let autoMaxRounds: number | undefined;
		let autoSummary = false;

		for (let i = 0; i < rawParts.length; i++) {
			const part = rawParts[i];
			if (part === "--extra") {
				const next = rawParts[i + 1];
				if (!next) {
					return { target: null, error: "Missing value for --extra" };
				}
				extraInstruction = next;
				i += 1;
				continue;
			}

			if (part.startsWith("--extra=")) {
				extraInstruction = part.slice("--extra=".length);
				continue;
			}

			if (part === "--auto") {
				autoRequested = true;
				continue;
			}

			if (part === "--summary") {
				autoSummary = true;
				continue;
			}

			if (part === "--review-model") {
				const next = rawParts[i + 1];
				if (!next) return { target: null, error: "Missing value for --review-model" };
				autoReviewModel = next;
				i += 1;
				continue;
			}

			if (part.startsWith("--review-model=")) {
				autoReviewModel = part.slice("--review-model=".length);
				continue;
			}

			if (part === "--max-rounds") {
				const next = rawParts[i + 1];
				if (!next) return { target: null, error: "Missing value for --max-rounds" };
				autoMaxRounds = Number.parseInt(next, 10);
				i += 1;
				continue;
			}

			if (part.startsWith("--max-rounds=")) {
				autoMaxRounds = Number.parseInt(part.slice("--max-rounds=".length), 10);
				continue;
			}

			parts.push(part);
		}

		if (autoMaxRounds !== undefined && (!Number.isFinite(autoMaxRounds) || autoMaxRounds < 1 || autoMaxRounds > 20)) {
			return { target: null, error: "--max-rounds must be between 1 and 20" };
		}

		if (!autoRequested && (autoReviewModel !== undefined || autoMaxRounds !== undefined || autoSummary)) {
			return {
				target: null,
				error: "--summary, --review-model, and --max-rounds require --auto",
			};
		}

		const auto: ParsedAutoFlags | undefined = autoRequested
			? { reviewModel: autoReviewModel, maxRounds: autoMaxRounds, summary: autoSummary }
			: undefined;

		const parsed = (target: ParsedReviewArgs["target"]): ParsedReviewArgs =>
			auto ? { target, extraInstruction, auto } : { target, extraInstruction };

		if (parts.length === 0) {
			return parsed(null);
		}

		const subcommand = parts[0]?.toLowerCase();

		switch (subcommand) {
			case "uncommitted":
				return parsed({ type: "uncommitted" });

			case "branch": {
				const branch = parts[1];
				if (!branch) return parsed(null);
				return parsed({ type: "baseBranch", branch });
			}

			case "commit": {
				const sha = parts[1];
				if (!sha) return parsed(null);
				// The title is read from `git log`, so trailing words are not a title any more.
				if (parts.length > 2) {
					return { target: null, error: `Unexpected arguments after ${sha}` };
				}
				return parsed({ type: "commitsList", shas: [sha] });
			}

			case "commits": {
				const shas = parts.slice(1);
				if (shas.length === 0) return parsed(null);
				return parsed({ type: "commitsList", shas });
			}

			case "after": {
				const sha = parts[1];
				if (!sha) return parsed(null);
				return parsed({ type: "commitsAfter", sha });
			}

			case "folder": {
				const paths = parseReviewPaths(parts.slice(1).join(" "));
				if (paths.length === 0) return parsed(null);
				return parsed({ type: "folder", paths });
			}

			case "pr": {
				const ref = parts[1];
				if (!ref) return parsed(null);
				return parsed({ type: "pr", ref });
			}

			default:
				return parsed(null);
		}
	}

	/**
	 * Handle PR checkout and return a ReviewTarget (or null on failure)
	 */
	async function handlePrCheckout(ctx: ExtensionContext, ref: string): Promise<ReviewTarget | null> {
		return await resolvePullRequestTarget(ctx, ref);
	}

	// Register the /review command
	pi.registerCommand("review", {
		description: "Review code changes (PR, uncommitted, branch, commits, or folder)",
		handler: async (args, ctx) => {
			if (!ctx.hasUI) {
				ctx.ui.notify("Review requires interactive mode", "error");
				return;
			}

			if (autoLoopRunning) {
				ctx.ui.notify("Auto review is already running. Use /review-auto-stop to stop it.", "warning");
				return;
			}

			// Check if we're already in a review
			if (reviewOriginId) {
				ctx.ui.notify("Already in a review. Use /end-review to finish first.", "warning");
				return;
			}

			// Check if we're in a git repository
			const { code } = await pi.exec("git", ["rev-parse", "--git-dir"]);
			if (code !== 0) {
				ctx.ui.notify("Not a git repository", "error");
				return;
			}

			// Try to parse direct arguments
			let target: ReviewTarget | null = null;
			let fromSelector = false;
			let extraInstruction: string | undefined;
			const parsed = parseArgs(args);
			if (parsed.error) {
				ctx.ui.notify(parsed.error, "error");
				return;
			}
			extraInstruction = parsed.extraInstruction?.trim() || undefined;

			if (parsed.target) {
				if (parsed.target.type === "pr") {
					// Handle PR checkout (async operation)
					target = await handlePrCheckout(ctx, parsed.target.ref);
					if (!target) {
						ctx.ui.notify("PR review failed. Returning to review menu.", "warning");
					}
				} else if (parsed.target.type === "commitsAfter") {
					// Resolve the range now so an empty range fails before a review starts.
					const { commits, error } = await getCommitsAfter(pi, parsed.target.sha);
					if (error) {
						ctx.ui.notify(`Failed to list commits after ${parsed.target.sha}: ${error}`, "error");
						return;
					}
					if (commits.length === 0) {
						ctx.ui.notify(
							`No commits found after ${parsed.target.sha}. Make sure the commit exists and is not HEAD.`,
							"error",
						);
						return;
					}
					target = { type: "commits", commits, rangeBase: parsed.target.sha };
				} else if (parsed.target.type === "commitsList") {
					const { commits, error } = await resolveCommitList(pi, parsed.target.shas);
					if (error) {
						ctx.ui.notify(`Failed to resolve commits: ${error}`, "error");
						return;
					}
					if (commits.length === 0) {
						ctx.ui.notify(`No commits found for: ${parsed.target.shas.join(" ")}`, "error");
						return;
					}
					target = { type: "commits", commits };
				} else {
					target = parsed.target;
				}
			}

			// If no args or invalid args, show selector
			if (!target) {
				fromSelector = true;
			}

			while (true) {
				if (!target && fromSelector) {
					target = await showReviewSelector(ctx);
				}

				if (!target) {
					ctx.ui.notify("Review cancelled", "info");
					return;
				}

				// Determine if we should use fresh session mode
				// Check if this is a new session (no messages yet)
				const entries = ctx.sessionManager.getEntries();
				const messageCount = entries.filter((e) => e.type === "message").length;

				// In an empty session, default to fresh review mode so /end-review works consistently.
				// Auto review always needs a review branch, so --auto forces fresh mode.
				let useFreshSession = messageCount === 0 || Boolean(parsed.auto);

				if (messageCount > 0 && !parsed.auto) {
					// Existing session - ask user which mode they want
					const choice = await ctx.ui.select("Start review in:", ["Empty branch", "Current session"]);

					if (choice === undefined) {
						if (fromSelector) {
							target = null;
							continue;
						}
						ctx.ui.notify("Review cancelled", "info");
						return;
					}

					useFreshSession = choice === "Empty branch";
				}

				let autoOptions: AutoReviewOptions | undefined;
				let reviewModel: ReviewModel | undefined;

				if (parsed.auto) {
					if (parsed.auto.reviewModel) {
						reviewModel = findModelByRef(ctx, parsed.auto.reviewModel);
						if (!reviewModel) {
							ctx.ui.notify(`Model not found: ${parsed.auto.reviewModel}`, "error");
							return;
						}
					}
					autoOptions = {
						reviewModel,
						maxRounds: parsed.auto.maxRounds ?? 5,
						summary: parsed.auto.summary,
						extraInstruction,
					};
				} else if (useFreshSession) {
					const after = await ctx.ui.select("After the review:", [
						"Review only (finish with /end-review)",
						"Auto: fix findings and re-review until clean",
					]);

					if (after === undefined) {
						if (fromSelector) {
							target = null;
							continue;
						}
						ctx.ui.notify("Review cancelled", "info");
						return;
					}

					if (after.startsWith("Auto")) {
						const modelChoice = await showReviewModelSelector(ctx);
						if (modelChoice === null) {
							ctx.ui.notify("Review cancelled", "info");
							return;
						}

						const maxRounds = await showAutoMaxRoundsSelector(ctx);
						if (maxRounds === null) {
							ctx.ui.notify("Review cancelled", "info");
							return;
						}

						autoOptions = {
							reviewModel: modelChoice.model,
							maxRounds,
							summary: false,
							extraInstruction,
						};
					} else {
						const modelChoice = await showReviewModelSelector(ctx);
						if (modelChoice === null) {
							ctx.ui.notify("Review cancelled", "info");
							return;
						}
						reviewModel = modelChoice.model;
					}
				}

				if (autoOptions) {
					await runAutoReviewLoop(ctx, target, autoOptions);
					return;
				}

				await executeReview(ctx, target, useFreshSession, { extraInstruction, reviewModel });
				return;
			}
		},
	});

	// Custom prompt for review summaries - focuses on preserving actionable findings
	const REVIEW_SUMMARY_PROMPT = `We are leaving a code-review branch and returning to the main coding branch.
Create a structured handoff that can be used immediately to implement fixes.

You MUST summarize the review that happened in this branch so findings can be acted on.
Do not omit findings: include every actionable issue that was identified.

Required sections (in order):

## Review Scope
- What was reviewed (files/paths, changes, and scope)

## Verdict
- "correct" or "needs attention"

## Findings
For EACH finding, include:
- Priority tag ([P0]..[P3]) and short title
- File location (\`path/to/file.ext:line\`)
- Why it matters (brief)
- What should change (brief, actionable)

## Fix Queue
1. Ordered implementation checklist (highest priority first)

## Constraints & Preferences
- Any constraints or preferences mentioned during review
- Or "(none)"

## Human Reviewer Callouts (Non-Blocking)
Include only applicable callouts (no yes/no lines):
- **This change adds a database migration:** <files/details>
- **This change introduces a new dependency:** <package(s)/details>
- **This change changes a dependency (or the lockfile):** <files/package(s)/details>
- **This change modifies auth/permission behavior:** <what changed and where>
- **This change introduces backwards-incompatible public schema/API/contract changes:** <what changed and where>
- **This change includes irreversible or destructive operations:** <operation and scope>

If none apply, write "- (none)".

These are informational callouts for humans and are not fix items by themselves.

Preserve exact file paths, function names, and error messages where available.`;

	const REVIEW_FIX_FINDINGS_PROMPT = `Use the latest review summary in this session and implement the review findings now.

Instructions:
1. Treat the summary's Findings/Fix Queue as a checklist.
2. Fix in priority order: P0, P1, then P2 (include P3 if quick and safe).
3. If a finding is invalid/already fixed/not possible right now, briefly explain why and continue.
4. Treat "Human Reviewer Callouts (Non-Blocking)" as informational only; do not convert them into fix tasks unless there is a separate explicit finding.
5. Follow fail-fast error handling: do not add local catch/fallback recovery unless this scope is an explicit boundary that can safely translate the failure.
6. If you add or keep a \`try/catch\`, explain the expected failure mode and either rethrow with context or return a boundary-safe error response.
7. JSON parsing/decoding should fail loudly by default; avoid silent fallback parsing.
8. Run relevant tests/checks for touched code where practical.
9. End with: fixed items, deferred/skipped items (with reasons), and verification results.`;

	type EndReviewAction = "returnOnly" | "returnAndFix" | "returnAndSummarize";
	type EndReviewActionResult = "ok" | "cancelled" | "error";
	type EndReviewActionOptions = {
		showSummaryLoader?: boolean;
		notifySuccess?: boolean;
	};

	function getActiveReviewOrigin(ctx: ExtensionContext): string | undefined {
		if (reviewOriginId) {
			return reviewOriginId;
		}

		const state = getReviewState(ctx);
		if (state?.active && state.originId) {
			reviewOriginId = state.originId;
			return reviewOriginId;
		}

		if (state?.active) {
			setReviewWidget(ctx, false);
			pi.appendEntry(REVIEW_STATE_TYPE, { active: false });
			ctx.ui.notify("Review state was missing origin info; cleared review status.", "warning");
		}

		return undefined;
	}

	function clearReviewState(ctx: ExtensionContext) {
		setReviewWidget(ctx, false);
		reviewOriginId = undefined;
		pi.appendEntry(REVIEW_STATE_TYPE, { active: false });
	}

	async function navigateWithSummary(
		ctx: ExtensionCommandContext,
		originId: string,
		showLoader: boolean,
	): Promise<{ cancelled: boolean; error?: string } | null> {
		if (showLoader && ctx.hasUI) {
			return ctx.ui.custom<{ cancelled: boolean; error?: string } | null>((tui, theme, _kb, done) => {
				const loader = new BorderedLoader(tui, theme, "Returning and summarizing review branch...");
				loader.onAbort = () => done(null);

				withReviewNavigation(() =>
					ctx.navigateTree(originId, {
						summarize: true,
						customInstructions: REVIEW_SUMMARY_PROMPT,
						replaceInstructions: true,
					}),
				)
					.then(done)
					.catch((err) => done({ cancelled: false, error: err instanceof Error ? err.message : String(err) }));

				return loader;
			});
		}

		try {
			return await withReviewNavigation(() =>
				ctx.navigateTree(originId, {
					summarize: true,
					customInstructions: REVIEW_SUMMARY_PROMPT,
					replaceInstructions: true,
				}),
			);
		} catch (error) {
			return { cancelled: false, error: error instanceof Error ? error.message : String(error) };
		}
	}

	async function executeEndReviewAction(
		ctx: ExtensionCommandContext,
		action: EndReviewAction,
		options: EndReviewActionOptions = {},
	): Promise<EndReviewActionResult> {
		// Read the state before navigating: the review-session entry stays on the review
		// branch, so the main-branch snapshot must be captured while still on it.
		const reviewState = getReviewState(ctx);
		const originId = getActiveReviewOrigin(ctx);
		if (!originId) {
			if (!getReviewState(ctx)?.active) {
				ctx.ui.notify("Not in a review branch (use /review first, or review was started in current session mode)", "info");
			}
			return "error";
		}

		const notifySuccess = options.notifySuccess ?? true;

		if (action === "returnOnly") {
			try {
				const result = await withReviewNavigation(() => ctx.navigateTree(originId, { summarize: false }));
				if (result.cancelled) {
					ctx.ui.notify("Navigation cancelled. Use /end-review to try again.", "info");
					return "cancelled";
				}
			} catch (error) {
				ctx.ui.notify(`Failed to return: ${error instanceof Error ? error.message : String(error)}`, "error");
				return "error";
			}

			clearReviewState(ctx);
			await restoreReviewMainSettings(ctx, reviewState);
			if (notifySuccess) {
				ctx.ui.notify("Review complete! Returned to original position.", "info");
			}
			return "ok";
		}

		const summaryResult = await navigateWithSummary(ctx, originId, options.showSummaryLoader ?? false);
		if (summaryResult === null) {
			ctx.ui.notify("Summarization cancelled. Use /end-review to try again.", "info");
			return "cancelled";
		}

		if (summaryResult.error) {
			ctx.ui.notify(`Summarization failed: ${summaryResult.error}`, "error");
			return "error";
		}

		if (summaryResult.cancelled) {
			ctx.ui.notify("Navigation cancelled. Use /end-review to try again.", "info");
			return "cancelled";
		}

		clearReviewState(ctx);
		await restoreReviewMainSettings(ctx, reviewState);

		if (action === "returnAndSummarize") {
			if (!ctx.ui.getEditorText().trim()) {
				ctx.ui.setEditorText("Act on the review findings");
			}
			if (notifySuccess) {
				ctx.ui.notify("Review complete! Returned and summarized.", "info");
			}
			return "ok";
		}

		pi.sendUserMessage(REVIEW_FIX_FINDINGS_PROMPT, { deliverAs: "followUp" });
		if (notifySuccess) {
			ctx.ui.notify("Review complete! Returned and queued a follow-up to fix findings.", "info");
		}
		return "ok";
	}

	async function runEndReview(ctx: ExtensionCommandContext): Promise<void> {
		if (!ctx.hasUI) {
			ctx.ui.notify("End-review requires interactive mode", "error");
			return;
		}

		if (autoLoopRunning) {
			requestAutoStop(ctx);
			return;
		}

		if (endReviewInProgress) {
			ctx.ui.notify("/end-review is already running", "info");
			return;
		}

		endReviewInProgress = true;
		try {
			const choice = await ctx.ui.select("Finish review:", [
				"Return only",
				"Return and fix findings",
				"Return and summarize",
			]);

			if (choice === undefined) {
				ctx.ui.notify("Cancelled. Use /end-review to try again.", "info");
				return;
			}

			const action: EndReviewAction =
				choice === "Return and fix findings"
					? "returnAndFix"
					: choice === "Return and summarize"
						? "returnAndSummarize"
						: "returnOnly";

			await executeEndReviewAction(ctx, action, {
				showSummaryLoader: true,
				notifySuccess: true,
			});
		} finally {
			endReviewInProgress = false;
		}
	}

	// Register the /end-review command
	pi.registerCommand("end-review", {
		description: "Complete review and return to original position",
		handler: async (_args, ctx) => {
			await runEndReview(ctx);
		},
	});

	pi.registerCommand("review-auto-stop", {
		description: "Stop the running auto review loop",
		handler: async (_args, ctx) => {
			if (!autoLoopRunning) {
				ctx.ui.notify("No auto review is running.", "info");
				return;
			}
			requestAutoStop(ctx);
		},
	});
}
