# Review rubric for specs and development plans

**What this is.** A checklist for reviewing a draft specification or development plan. Every check is a yes/no question. The checks come from published guidance on requirements, design documents, specs for AI coding agents, and plain-language writing. Each one cites the source it came from.

**Who it is for.** Reviewers of the development plan for the new spectrum fitter, the Python tool zeemap. The plan has three parts: (1) build specifications, (2) verification loops, (3) quality gates for pushing to `main`. Both humans and AI agents can apply the rubric.

**How to use it.**
1. Read the draft once as a first-time reader, before you open this rubric.
2. Answer every check with *yes*, *no* or *n/a*. For each *no*, write one sentence that says where the problem is.
3. A *no* on any of the ten checks marked **[TOP 10]** blocks acceptance. Any other *no* is a request for a change.
4. Then scan the draft for the anti-patterns at the end of this file.

**Status.** Written 2026-10-17 by a research agent. Every source listed was opened on that date unless it is marked [unverified]. Section numbers refer to the source as opened.

**Terms used here.**
- *Requirement*: one statement of what the software must do or what property it must have.
- *Acceptance criterion*: a condition, checkable by a test, that shows a requirement is met.
- *Spec*: the document that holds the requirements. *Plan*: the document that says how and in what order the software gets built and checked.
- *Agent*: an AI coding agent (for example Claude Code) that implements tasks from the spec.
- *Oracle*: an independent source of the right answer that a test compares against, such as a reference output, an analytic solution or a published fit.

---

## The 10 most important checks

| # | Check | Group |
|---|---|---|
| 1 | A1: Does every requirement state how it will be verified? | Requirement quality |
| 2 | A2: Is every requirement free of vague words, with numbers, units and tolerances where a quantity is meant? | Requirement quality |
| 3 | A3: Do the main behaviours come with concrete worked examples (inputs and expected outputs)? | Requirement quality |
| 4 | A4: Are open questions marked as open instead of being silently decided? | Requirement quality |
| 5 | B1: Does the first paragraph say what is being built, why, and what state the plan is in? | Structure |
| 6 | B2: Are there explicit non-goals? | Structure |
| 7 | C1: Does every task end with a runnable check that returns pass or fail? | Agent-executable |
| 8 | C2: Does every task or spec section name the files, interfaces and out-of-scope items, and end with an end-to-end check? | Agent-executable |
| 9 | D1: Is every specialist term and abbreviation explained where it first appears? | Readability |
| 10 | D2: Has a real first-time reader paraphrased the draft back correctly? | Readability |

---

## A. Requirement quality

Ranked by importance within the group.

### A1 [TOP 10]. Every requirement can be verified, and the spec says how
- **Practice.** Write each requirement so that a test, a script or an inspection can show whether it is met, and name that method next to the requirement.
- **Why.** ISO/IEC/IEEE 29148:2018 lists *verifiable* as a required property of every individual requirement. Kiro gives as a benefit of its requirement format: "Each requirement can be directly translated into test cases". Spec Kit's readiness checklist includes "Requirements are testable and unambiguous" and measurable success criteria.
- **Sources.** ISO/IEC/IEEE 29148:2018, characteristics of individual requirements, read through the secondary summary at https://www.modernrequirements.com/blogs/iso-29148-explained/ (the standard is paywalled and was not opened). Kiro feature specs, https://kiro.dev/docs/specs/feature-specs/ . GitHub Spec Kit, `spec-driven.md`, https://github.com/github/spec-kit/blob/main/spec-driven.md .
- **Reviewer check.** *Does every requirement have a stated way to verify it (test name, script, comparison against a reference, or inspection)?*

### A2 [TOP 10]. No vague words; quantities have numbers, units and tolerances
- **Practice.** Replace words like "fast", "accurate", "robust", "reasonable" or "as appropriate" with a measurable statement. For example, write "recovers the injected polar field B_p within 2 % for S/N >= 50" instead of "recovers the field accurately".
- **Why.** The 29148 summary gives "user-friendly" as an example of a non-verifiable term. Femmer et al. derive "requirements smells" from 29148: subjective language, ambiguous adverbs and adjectives, loopholes, open-ended non-verifiable terms, superlatives, comparatives without a baseline, and vague pronouns. In their evaluation, automatic smell detection reached about 59 % precision and 82 % recall.
- **Sources.** Femmer, Méndez Fernández, Wagner, Eder, "Rapid quality assurance with Requirements Smells", *J. Syst. Softw.* (journal reference 2016 on arXiv), DOI 10.1016/j.jss.2016.02.047, https://arxiv.org/abs/1611.08847 . The 29148 secondary summary above.
- **Reviewer check.** *Is every requirement free of subjective or comparative words that have no baseline, with a number, a unit and a tolerance wherever a quantity is meant?*

### A3 [TOP 10]. Key examples illustrate the main behaviours
- **Practice.** For each main behaviour, give at least one concrete example with inputs and the expected outcome. Use Given-When-Then where it fits: *Given* a synthetic spectrum made from a known dipole at B_p = 300 MG, *when* the fitter runs with default settings, *then* the best fit recovers B_p within tolerance.
- **Why.** Fowler: examples "are usually much easier to come up with" than general pre- and postconditions. Examples can also be checked directly, because they become tests and so give a second, independent statement of the behaviour. Given-When-Then splits a scenario into precondition, behaviour and expected outcome. Spec Kit's spec template asks for acceptance scenarios in that form.
- **Sources.** Fowler, "Specification By Example" (2004), https://martinfowler.com/bliki/SpecificationByExample.html . Fowler, "GivenWhenThen" (2013), https://martinfowler.com/bliki/GivenWhenThen.html . Spec Kit spec template, https://github.com/github/spec-kit/blob/main/templates/spec-template.md . Adzic, *Specification by Example* (Manning, 2011), https://gojko.net/books/specification-by-example/ (only the book page was opened; the book's patterns are [unverified]).
- **Reviewer check.** *Does every main behaviour have at least one worked example with concrete inputs and expected outputs?*

### A4 [TOP 10]. Open questions are marked, not guessed
- **Practice.** When something is undecided, mark it in place, for example `[NEEDS CLARIFICATION: which spectral resolution does the 2026 library use?]`, and collect these markers in an "Open questions" section. The spec is not ready while markers remain.
- **Why.** Spec Kit says its markers prevent "the common LLM behavior of making plausible but potentially incorrect assumptions", and its readiness gate requires that no markers remain. PEP 1 asks for "Open Issues", and the Rust RFC template asks for "Unresolved questions".
- **Sources.** Spec Kit `spec-driven.md` (above). PEP 1, "What belongs in a successful PEP", https://peps.python.org/pep-0001/ . Rust RFC template, https://github.com/rust-lang/rfcs/blob/master/0000-template.md .
- **Reviewer check.** *Is every undecided point marked in place and listed in an open-questions section, and does the plan forbid implementing a requirement that still has a marker?*

### A5. One requirement per statement
- **Practice.** Split statements that join several demands with "and", "or" or "as well as".
- **Why.** 29148 requires each requirement to be *singular*, defining one aspect only, so that it can be traced and tested on its own.
- **Source.** 29148 secondary summary (above).
- **Reviewer check.** *Does each numbered requirement state exactly one thing, so that a test can fail for exactly one reason?*

### A6. Stable IDs and traceability to goals
- **Practice.** Give each requirement an ID (FR-001 for functional requirements, SC-001 for success criteria) and link it to a goal. Tasks and tests then cite these IDs.
- **Why.** Spec Kit uses "FR-001: System MUST ..." and SC-nnn success criteria. Kiro lists "Traceability: Individual requirements can be tracked through implementation" as a benefit (that each Kiro task cites requirement IDs is [unverified]). 29148 requires each requirement to be *necessary*: removing it leaves a gap.
- **Sources.** Spec Kit spec template; Kiro feature specs; 29148 summary (all above).
- **Reviewer check.** *Does every requirement have a stable ID and a link to a stated goal, and could you delete any requirement without losing something a goal needs?*

### A7. Conditions, triggers and failure cases follow an explicit pattern
- **Practice.** Write conditional requirements with the EARS patterns: *While* (state), *When* (event), *Where* (optional feature), *If ... then* (unwanted behaviour), or a combination. Write the failure cases out in full. For example: "If the library does not cover the requested Teff, then the fitter MUST stop with an error that names the covered range."
- **Why.** EARS (Mavin et al. 2009, developed at Rolls-Royce) is a small set of rules that addresses eight common requirement problems, among them ambiguity, complexity and vagueness. Kiro writes its acceptance criteria as "WHEN [condition/event] THE SYSTEM SHALL [expected behavior]". Spec Kit's template has a separate edge-case section ("What happens when ...?").
- **Sources.** Mavin, Wilkinson, Harwood, Novak, "Easy Approach to Requirements Syntax (EARS)", RE'09, pp. 317-322, DOI 10.1109/RE.2009.9 (record at https://research.manchester.ac.uk/en/publications/easy-approach-to-requirements-syntax-ears/). Pattern list at https://alistairmavin.com/ears/ . Kiro feature specs; Spec Kit template (above).
- **Reviewer check.** *Is every conditional or error behaviour written with an explicit trigger or state (When / While / If-then), and does every input the user can get wrong have a stated response?*

### A8. Normative keywords are used deliberately and consistently
- **Practice.** Choose one convention and say which in the document. Either use BCP 14 keywords in capitals (MUST, SHOULD, MAY), with the RFC 8174 sentence that defines them, or use plain "must" and "should". Never use "shall" and "must" together, and use the keywords sparingly.
- **Why.** RFC 8174 says the keywords carry their special meaning "only when they are in all capitals". RFC 2119 section 6 says the imperatives "must be used with care and sparingly" and must not be used to impose a method that is not required. The plain-language guidelines say "shall" is ambiguous and that "must" and "shall" should not appear in the same text.
- **Sources.** RFC 2119, section 6, https://www.rfc-editor.org/rfc/rfc2119 . RFC 8174, https://www.rfc-editor.org/rfc/rfc8174 . Federal Plain Language Guidelines, "Shall and must", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/conversational/shall-and-must.md .
- **Reviewer check.** *Does the document say how MUST, SHOULD and MAY are meant, use them only for real obligations, and avoid "shall"?*

### A9. Requirements say *what* and *why*, not *how*, unless the how is a real constraint
- **Practice.** Keep requirements free of implementation choices. Put design choices in the plan or design section, with reasons. When a method really is required, for example "MUST read the existing MAGWD `GRID.OUT` format", say why.
- **Why.** Spec Kit's template tells writers to "Focus on WHAT users need and WHY". RFC 2119 section 6 warns against using imperatives to impose a method when the method is not required.
- **Sources.** Spec Kit `spec-driven.md`; RFC 2119 section 6 (above).
- **Reviewer check.** *Is every implementation detail in a requirement either removed or justified as a real constraint?*

### A10. Quality requirements are written as measurable scenarios
- **Practice.** Write non-functional needs as scenarios with a stimulus, the environment, the expected response and a response measure. For example: "Fitting one ph16-sized spectrum against the standard library on a laptop takes under N minutes."
- **Why.** arc42 section 10.2: "Quality scenarios make quality requirements concrete and allow to decide whether they are fulfilled (in the sense of acceptance criteria)"; scenarios must be "specific and measurable".
- **Source.** arc42 section 10, https://docs.arc42.org/section-10/ .
- **Reviewer check.** *Is every speed, memory, reproducibility or usability requirement written as a scenario with a measurable response?*

### A11. The requirement set is consistent
- **Practice.** Check that no two requirements contradict each other, and that tolerances agree between the spec, the verification loops and the gates.
- **Why.** 29148 requires the requirement *set* to be complete, consistent, feasible, comprehensible and validatable.
- **Source.** 29148 secondary summary (above).
- **Reviewer check.** *Are the same quantities (tolerances, parameter ranges, file formats) stated identically wherever they appear across the three parts of the plan?*

---

## B. Document structure

### B1 [TOP 10]. The bottom line comes first
- **Practice.** The first paragraph says what is being built, for whom, why, and what state the document is in (draft, accepted, superseded). Details follow in order of decreasing importance.
- **Why.** GOV.UK: "Put the most important information first", tapering to details, "sometimes called an 'inverted pyramid' approach". Its guidance also notes that most people read only 20 to 28 % of the text on a web page. Google Technical Writing: "Imagine that your peers might only read the first paragraph of your document." BLUF ("bottom line up front") is the US Army name for the same rule; AR 25-50 is cited as its origin [unverified: the regulation was not opened].
- **Sources.** GOV.UK, "Create a clear structure for your content", https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-structure/ . Google Technical Writing One, "Documents", https://developers.google.com/tech-writing/one/documents .
- **Reviewer check.** *Could a reader who stops after the first paragraph say what is being built, why, and whether the plan is final?*

### B2 [TOP 10]. Goals and non-goals are explicit
- **Practice.** List the goals. Then list the non-goals: things a reader could reasonably expect the project to do but that it deliberately leaves out. For example: "Non-goal: radiative transfer; the fitter uses a precomputed library."
- **Why.** Ubl: non-goals are "things that could reasonably be goals, but are explicitly chosen not to be goals". Google Technical Writing: a good document also states its "non-scope", and the non-scope should list only what readers "would reasonably expect". Claude Code's guidance: the most useful specs "state what is out of scope".
- **Sources.** Ubl, "Design Docs at Google", https://www.industrialempathy.com/posts/design-docs-at-google/ . Google Technical Writing One, "Documents" (above). Claude Code best practices, "Let Claude interview you", https://code.claude.com/docs/en/best-practices .
- **Reviewer check.** *Does the document list non-goals, and is each one something a reasonable reader might otherwise have expected?*

### B3. The audience and prerequisites are stated
- **Practice.** Say who the document is for and what each reader is expected to know already. For example: "Assumes you know what a dipole field geometry is. Read `PRIMER.md` first."
- **Why.** Google Technical Writing: "A good document explicitly specifies its audience", including prerequisite knowledge and prerequisite reading. Its definition is "good documentation = knowledge and skills your audience needs to do a task − your audience's current knowledge and skills".
- **Source.** Google Technical Writing One, "Documents" and "Audience", https://developers.google.com/tech-writing/one/audience .
- **Reviewer check.** *Does the document name its readers (for example researcher, collaborator, student, agent) and the background each one needs?*

### B4. Context and motivation come before the design
- **Practice.** Before the design, give objective background: the problem, what existing practice does, and why a new tool is needed.
- **Why.** Ubl: the context section "should be entirely focused on objective background facts". PEP 1 calls the motivation "critical" and says it "should clearly explain why the existing ... is inadequate". The Rust RFC template has a Motivation section.
- **Sources.** Ubl; PEP 1; Rust RFC template (above).
- **Also.** Google Technical Writing: "compare and contrast your ideas with concepts that your audience already understands" ("Documents").
- **Reviewer check.** *Does the document explain, with facts rather than opinions, why existing practice is not enough?*

### B5. Alternatives and rejected ideas are recorded with reasons
- **Practice.** List the realistic alternatives, for example a different optimiser or sampler, together with the trade-offs and why each was rejected.
- **Why.** Ubl: this section shows "why the selected solution is the best given the project goals". PEP 1: rejected ideas "should be recorded along with the reasoning". The Rust RFC template has "Rationale and alternatives" and "Drawbacks" ("Why should we *not* do this?").
- **Sources.** Ubl; PEP 1; Rust RFC template (above).
- **Reviewer check.** *For each major design choice, is at least one real alternative named, with the reason it lost?*

### B6. Decisions are logged as short records with consequences
- **Practice.** Record each significant decision as a short architecture decision record (ADR) of one or two pages: context, decision ("We will ..."), status, and consequences, including the negative ones. Keep superseded records and link them to the records that replace them.
- **Why.** Nygard: without such records, "a new person coming on to a project may be perplexed, baffled, delighted, or infuriated by some past decision" and is left to "blindly accept" or "blindly change" it. He also writes "Large documents are never kept up to date. Small, modular documents have at least a chance at being updated", and recommends writing each record "as if it is a conversation with a future developer".
- **Source.** Nygard, "Documenting Architecture Decisions" (2011), https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions .
- **Also.** PEP 1 asks for a change history, and Ubl says a design doc should be updated during implementation while the system has not shipped yet.
- **Reviewer check.** *Is each significant decision recorded with its status and consequences (including the negative ones), with superseded decisions kept and linked, and is there a change history for the plan itself?*

### B7. Different kinds of content are kept apart
- **Practice.** Keep the normative spec (reference: what must be true), the rationale (explanation: why), the how-to material (for example how to run the gates) and any tutorial (a first walk-through for a new student) in separate sections or files, and link between them.
- **Why.** Diátaxis names four kinds of documentation (tutorials, how-to guides, reference, explanation) that serve different needs, and says "crossing or blurring the boundaries ... is at the heart of a vast number of problems in documentation". Reference material should be "free of distraction and interpretation".
- **Source.** Diátaxis, "Start here", https://diataxis.fr/start-here/ .
- **Reviewer check.** *Can a reader find the binding requirements without reading rationale, and the rationale without wading through requirements?*

### B8. The length fits the readers
- **Practice.** Keep each document short enough for busy readers to finish. Split the plan into modular files (spec, verification, gates, decisions) instead of writing one large document.
- **Why.** Ubl: design docs should be "sufficiently detailed but short enough to actually be read by busy people", about 10-20 pages for a large project and 1-3 pages for a mini design doc. Nygard: small, modular documents get updated. Claude Code's guidance warns that over-long instruction files make "important rules get lost in the noise".
- **Sources.** Ubl; Nygard; Claude Code best practices (above).
- **Reviewer check.** *Is each document short enough to read in one sitting, with nothing that the stated scope does not need?*

### B9. A guide-level explanation teaches the result as if it already existed
- **Practice.** Include one section that shows a new user how to fit a spectrum with the finished tool, written as if the tool already worked.
- **Why.** The Rust RFC template asks for an explanation written "as if it was already included in the language and you were teaching it to another Rust programmer". PEP 1 has a "How to Teach This" section.
- **Sources.** Rust RFC template; PEP 1 (above).
- **Reviewer check.** *Is there a short walk-through from a new user's point of view showing the input, the command and the output?*

---

## C. Specs an agent can execute

### C1 [TOP 10]. Every task ends with a runnable check that returns pass or fail
- **Practice.** End each task with the exact command that shows it is done, for example `pytest tests/test_dipole.py -k recover_bp`, and the result that counts as passing. Prose descriptions of what a working result looks like do not count.
- **Why.** Claude Code's guidance: "Claude stops when the work looks done. Without a check it can run, 'looks done' is the only signal available, and you become the verification loop". A check is "anything that returns a signal Claude can read": a test suite, a build exit code, a linter, or "a script that diffs output against a fixture". Its list of common failure patterns includes "the trust-then-verify gap", with the fix: "If you can't verify it, don't ship it."
- **Sources.** Claude Code best practices, "Give Claude a way to verify its work", https://code.claude.com/docs/en/best-practices . Claude Code best practices, "Avoid common failure patterns".
- **Reviewer check.** *Does every task end with a command that a person or an agent can run, and a stated pass condition?*

### C2 [TOP 10]. Each spec unit is self-contained and ends with an end-to-end check
- **Practice.** Each spec section or task names the files and interfaces involved, states what is out of scope, and ends with one end-to-end step that proves the feature works: for example, fit the ph16 observation and compare the result with the stored reference fit.
- **Why.** Claude Code's guidance: "The most useful specs are self-contained: they name the files and interfaces involved, state what is out of scope, and end with an end-to-end verification step that proves the feature works." Its prompt examples contrast vague requests ("add tests for foo.py") with scoped ones that name the file, the scenario and the testing preference.
- **Source.** Claude Code best practices, "Let Claude interview you" and "Provide specific context in your prompts" (above).
- **Reviewer check.** *Could an agent starting in a fresh session do this task from the task text and the files it names, without asking what is in scope?*

### C3. The oracle is named for every scientific result
- **Practice.** For each numerical claim, name what it is compared against and at what tolerance: synthetic spectra with injected parameters, the published fits (Kulebi et al. 2009, RE J0317-853), or analytic limits such as B -> 0. Every bug that gets fixed becomes a test.
- **Why.** Wilson et al., practice 5 "Plan for mistakes": "Add assertions to programs to check their operation", "Use an off-the-shelf unit testing library", "Turn bugs into test cases". Practice 3 is "Work in small steps with frequent feedback and course correction". The choice of oracles listed above is this project's own, not taken from the paper.
- **Source.** Wilson et al., "Best Practices for Scientific Computing", *PLoS Biol* 12(1): e1001745 (2014), https://doi.org/10.1371/journal.pbio.1001745 .
- **Reviewer check.** *Does every verification loop name its reference (synthetic truth, published fit, legacy-code output or analytic limit) and its tolerance, and say how a fixed bug becomes a test?*

### C4. Tasks are small, trace to requirement IDs, and state their dependencies
- **Practice.** Break the work into tasks that one session can finish. Each task cites the requirement IDs it serves and lists the tasks it depends on, so that independent tasks can run in parallel.
- **Why.** Kiro "builds a dependency graph of the tasks in your tasks.md and groups independent tasks into waves" that run concurrently. Spec Kit marks independent tasks `[P]` ("Parallelization"). Spec Kit's checklist items cite spec sections such as "[Spec §FR-1]". Osmani: dividing the work into focused tasks keeps output quality up, and quality drops "as you pile on more instructions or data into the prompt".
- **Sources.** Kiro specs, https://kiro.dev/docs/specs/ . Spec Kit `spec-driven.md`. Osmani, "How to write a good spec for AI agents" (2026-01-13), https://addyosmani.com/blog/good-spec/ .
- **Reviewer check.** *Does every task cite at least one requirement ID, list its dependencies, and fit in a single working session?*

### C5. Exact commands appear early
- **Practice.** Near the top, list the exact build, test, lint and gate commands with their flags, for example `tools/check.sh 1 --fast`.
- **Why.** Nigh (GitHub, analysis of more than 2,500 `agents.md` files): effective files "put relevant executable commands in an early section". Claude Code's guidance lists "Bash commands Claude can't guess" and "testing instructions and preferred test runners" as things to include.
- **Sources.** Nigh, "How to write a great agents.md: lessons from over 2,500 repositories" (GitHub Blog, 2025-11-19), https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/ . Claude Code best practices, "Write an effective CLAUDE.md".
- **Keep the standing file short.** Claude Code: "For each line, ask: 'Would removing this cause Claude to make mistakes?' If not, cut it. Bloated CLAUDE.md files cause Claude to ignore your actual instructions!" Nigh: "Most agent files fail because they're too vague."
- **Reviewer check.** *Are the build, test and gate commands listed verbatim near the start of the fitter repo's agent instructions, do they run as written, and is every other line of those instructions one whose removal would cause a mistake?*

### C6. Boundaries are stated in three tiers
- **Practice.** Say what the agent always does, what it asks about first, and what it never does. For this project, *never* includes: copying source of Numerical Recipes, other fitting programs or MAGWD into the repository, and merging its own change.
- **Why.** Osmani recommends a three-tier boundary system ("Always do", "Ask first", "Never do"). Nigh: "Tell AI what it should never touch"; "'Never commit secrets' was the most common helpful constraint."
- **Sources.** Osmani; Nigh (above).
- **Reviewer check.** *Does the plan list explicit always, ask-first and never rules, including the licence and IP limits?*

### C7. Tests are written and seen to fail before the code they test
- **Practice.** For each behaviour, write the test first, confirm that it fails, then implement. For bug fixes, first write a test that reproduces the bug.
- **Why.** Spec Kit's constitution requires that tests are written, approved and "confirmed to FAIL" before any implementation code. Claude Code's guidance gives "write a failing test that reproduces the issue, then fix it" as an example of a good prompt.
- **Sources.** Spec Kit `spec-driven.md`; Claude Code best practices, "Provide specific context in your prompts".
- **Reviewer check.** *Does the plan require each task's test to exist, and to fail, before the implementation is written?*

### C8. An independent review checks the result against the spec
- **Practice.** Before a task counts as done, a reviewer working in a fresh context, either a person or a subagent, checks the diff against the spec: every requirement implemented, listed edge cases tested, nothing outside scope changed. Only gaps that affect correctness or the stated requirements count as findings.
- **Why.** Claude Code's guidance: such a reviewer "sees only the diff and the criteria you give it, not the reasoning that produced the change". It also warns that "chasing every finding leads to over-engineering", so reviewers should flag only gaps that affect correctness or requirements.
- **Source.** Claude Code best practices, "Add an adversarial review step".
- **Reviewer check.** *Does every gate include a review against the spec by someone (or something) that did not write the change, with a stated rule for what counts as a finding?*

### C9. Point to existing patterns and real examples, not prose descriptions
- **Practice.** Point the agent at concrete files to follow, for example "read the library format the way `src/zeemap/library.py` does", and show code or data snippets instead of describing them.
- **Why.** Claude Code's guidance: "Reference existing patterns. Point Claude to patterns in your codebase." Nigh lists "Code examples over explanations" among what the best-performing agent files do. Osmani: "One real code snippet showing your style beats three paragraphs describing it."
- **Sources.** Claude Code best practices; Nigh; Osmani (above).
- **Reviewer check.** *Where the plan describes a format, interface or style, does it link to a real file or snippet that shows it?*

### C10. The spec itself is checked like code before implementation starts
- **Practice.** Run this rubric, or a project-specific checklist built from it, over the spec before any task starts. The checklist items test the writing, not the software. Ask "Is the tolerance for B_p stated?", not "Does the fit recover B_p?".
- **Why.** Spec Kit's checklist command calls checklists "UNIT TESTS FOR REQUIREMENTS WRITING", checking completeness, clarity, consistency, measurability and coverage. It contrasts wrong items ("Verify landing page displays 3 episode cards") with right ones ("Are the number and layout of featured episodes explicitly specified?").
- **Source.** Spec Kit, `templates/commands/checklist.md`, https://github.com/github/spec-kit/blob/main/templates/commands/checklist.md .
- **Reviewer check.** *Was the spec reviewed against a writing-quality checklist, with all findings closed, before implementation began?*

---

## D. Readability and empathy for first-time readers

### D1 [TOP 10]. Every specialist term and abbreviation is explained at first use
- **Practice.** Explain each specialist term in plain words where it first appears, for example "the cyclotron resonance, the wavelength where free electrons gyrating in the field absorb strongly". When there are many terms, also collect them in a glossary. Prefer a short name to an abbreviation ("the library", not "ML"), and keep the number of new abbreviations small.
- **Why.** GOV.UK: specialist terms are acceptable, "You just need to explain what they mean the first time you use them". Google Technical Writing: define new terms, and "if your document is introducing many terms, collect the definitions into a glossary". The plain-language guidelines suggest at most two or three abbreviations in a document, with nicknames instead of the rest.
- **Sources.** GOV.UK, "Use clear language", https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-language/ . Google Technical Writing One, "Words", https://developers.google.com/tech-writing/one/words . Plain Language Guidelines, "Minimize abbreviations", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/words/minimize-abbreviations.md .
- **Reviewer check.** *Is every specialist term, symbol and abbreviation explained where it first appears, with a glossary when there are more than a handful?*

### D2 [TOP 10]. A real first-time reader has tested the draft
- **Practice.** Before accepting the draft, give it to one or more people who were not involved, for example a student or a collaborator. Ask them to explain each section back in their own words. Wherever their explanation differs from what was meant, fix the text. Do not count "Is it clear?" answered yes as a test.
- **Why.** The curse of knowledge: experts "forget that novices don't know what you already know" (Google Technical Writing). Heath and Heath describe how experts speak in abstractions that summarise knowledge only they hold. Paraphrase testing, from the plain-language guidelines, uses 6 to 9 one-on-one interviews, and the guidelines say to avoid yes/no questions.
- **Sources.** Google Technical Writing One, "Audience" (curse of knowledge). Heath and Heath, "The Curse of Knowledge", *Harvard Business Review*, December 2006, https://hbr.org/2006/12/the-curse-of-knowledge (only the summary paragraph was opened). Plain Language Guidelines, "Paraphrase testing", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/test/paraphrase-testing.md . The "tappers and listeners" experiment often cited for this effect is [unverified].
- **Reviewer check.** *Has at least one person outside the project explained the draft back correctly in their own words, and were the places they misread fixed?*

### D3. One name for one thing
- **Practice.** Use the same term for the same concept throughout all three parts of the plan. If the text says "library", it does not switch to "grid", "model set" or "bank".
- **Why.** Plain-language guidelines: different terms for one concept make readers "wonder if you're referring to the same group". Google Technical Writing: "if you rename a term in the middle of a document, your ideas won't compile (in your users' heads)".
- **Sources.** Plain Language Guidelines, "Use the same terms consistently", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/words/use-the-same-terms-consistently.md . Google Technical Writing One, "Words".
- **Reviewer check.** *Is each concept called by exactly one name across all documents, and does that name match the glossary?*

### D4. Sentences and paragraphs are short
- **Practice.** One idea per sentence. Split sentences longer than about 25 words. Keep paragraphs to five sentences or fewer. Use lists for series of items.
- **Why.** GOV.UK: "Try to split up sentences that are over 25 words long. Paragraphs should have no more than 5 sentences each." Plain-language guidelines: "Express only one idea in each sentence." Google Technical Writing: short sentences "are usually easier to understand than long sentences".
- **Sources.** GOV.UK, "Use clear language". Plain Language Guidelines, "Write short sentences", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/concise/write-short-sentences.md . Google Technical Writing One, "Short sentences", https://developers.google.com/tech-writing/one/short-sentences .
- **Reviewer check.** *Is every sentence over 25 words justified, and does every paragraph hold one idea?*

### D5. Active voice with a named actor
- **Practice.** Say who does what: "the fitter writes ...", "the reviewer runs ...", "we will ...". Avoid "it must be done".
- **Why.** Plain-language guidelines: "Active voice makes it clear who should do what. It eliminates ambiguity about responsibilities. Not 'It must be done,' but 'You must do it.'" Nygard writes decisions in "full sentences, with active voice. 'We will ...'".
- **Sources.** digital.gov, "Writing for understanding", https://digital.gov/guides/plain-language/writing . Nygard (above).
- **Reviewer check.** *Does every obligation name who carries it out (the code, the agent, the reviewer or a named person)?*

### D6. Headings describe the content
- **Practice.** Write headings that say what the section contains, for example "Recover injected field parameters from synthetic spectra" rather than "Verification", and that put the important word first. The text should still make sense if the headings are removed.
- **Why.** GOV.UK: headings should be "descriptive – avoid generic headings like 'Introduction'", "frontloaded", "active" and "removable", and should not use technical terms that have not been explained yet.
- **Source.** GOV.UK, "Create a clear structure for your content" (above).
- **Reviewer check.** *Can a reader skimming only the headings tell what each section requires?*

### D7. The main rule comes before exceptions and conditions
- **Practice.** State the rule first and the exceptions after it. Avoid opening a sentence with "Except as described in ...".
- **Why.** Plain-language guidelines: a sentence that opens with an exception makes the reader absorb "the exception, then the rule", and the reader usually has to reread. "In general, the main point of the sentence should be as close to the beginning as possible."
- **Source.** Plain Language Guidelines, "Place the main idea before exceptions and conditions", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/organize/place-the-main-idea-before-exceptions-and-conditions.md .
- **Reviewer check.** *Does every rule that has exceptions state the rule first?*

### D8. Explanatory worked examples for hard concepts
- **Practice.** Where a concept is hard (field geometry parameters, the χ² surface, library interpolation), walk through one concrete case with real numbers.
- **Why.** Plain-language guidelines: "Good examples can substitute for long explanations. The more complex the concept you are writing about, the more you should consider using an example." They also recommend writing "for example" instead of "e.g.".
- **Source.** Plain Language Guidelines, "Use examples", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/conversational/use-examples.md .
- **Reviewer check.** *Does each concept a student would find hard come with a worked numerical example?*

### D9. Conditional material goes in tables
- **Practice.** Put "if this, then that" material in a table: which gate runs for which change, which tolerance applies to which comparison.
- **Why.** Plain-language guidelines: "Probably the most useful type of table is the 'if-then table'", and tables "help your audience see relationships that are often times hidden in dense text".
- **Source.** Plain Language Guidelines, "Use tables to make complex material easier to understand", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/design/use-tables-to-make-complex-material-easier-to-understand.md .
- **Reviewer check.** *Is every rule of the form "if X, then run or use Y" given as a table rather than as prose?*

### D10. Plain words over jargon; common words keep their ordinary meaning
- **Practice.** Use the necessary technical terms, but write the surrounding text plainly. Do not redefine everyday words with special meanings (for example, do not use "pass" for a fit that converged).
- **Why.** Plain-language guidelines: "Readers complain about jargon more than any other writing fault, because writers often fail to realize that terms they know well may be difficult or meaningless to their audience". They also warn never to define a word "to mean something other than its commonly accepted meaning", because readers forget the special meaning.
- **Sources.** Plain Language Guidelines, "Avoid jargon", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/words/avoid-jargon.md , and "Minimize definitions", https://github.com/GSA/plainlanguage.gov/blob/main/_pages/guidelines/words/minimize-definitions.md .
- **Reviewer check.** *Does every everyday word keep its ordinary meaning, with jargon used only where no plain word exists?*

---

## Anti-patterns: what specs commonly get wrong

Scan the draft for each of these. Every item names its source.

| # | Anti-pattern | What it looks like here | Source |
|---|---|---|---|
| 1 | **Unverifiable wording** | "The fitter should be accurate and robust." | 29148 (*verifiable*, *unambiguous*; secondary summary); Femmer et al., requirements smells (subjective language, ambiguous adjectives, open-ended terms) |
| 2 | **Comparatives and superlatives with no baseline** | "Faster than the old code", "the best geometry". Faster on which spectrum and machine, and by how much? | Femmer et al. (comparatives, superlatives) |
| 3 | **Bundled requirements** | "Reads the library and interpolates in Teff and B and reports errors." One test cannot fail cleanly. | 29148 *singular* |
| 4 | **"Looks done" as the finish line** | A task with no command and no pass condition, so the agent stops when the code looks plausible. | Claude Code best practices, "the trust-then-verify gap" |
| 5 | **Silent assumptions** | The agent picks a wavelength grid or a prior because the spec did not say, and nobody notices. | Spec Kit `spec-driven.md` ("plausible but potentially incorrect assumptions") |
| 6 | **An implementation manual with no reasons** | Steps without trade-offs or rejected alternatives, so a later reader cannot tell what may change. | Ubl, "Design Docs at Google" (when not to write a design doc); Nygard (blindly accept or blindly change) |
| 7 | **The monolith that is never updated** | One long plan that drifts from the code within weeks. | Nygard ("Large documents are never kept up to date") |
| 8 | **The bloated standing instruction file** | An `AGENTS.md` so long that the rules that matter get lost. | Claude Code best practices, "The over-specified CLAUDE.md"; Nigh ("too vague") |
| 9 | **Overused or mixed keywords** | MUST on style preferences; "shall" and "must" in the same document; lower-case "must" read as normative. | RFC 2119 section 6; RFC 8174; Plain Language "Shall and must" |
| 10 | **Mixed document kinds** | Requirements buried in tutorial prose, or a reference table full of opinion. | Diátaxis ("crossing or blurring the boundaries") |
| 11 | **Abbreviation soup and undefined terms** | "The CP of the q=+1 core under RS with ALI", with none of it explained. | Plain Language "Minimize abbreviations"; GOV.UK "Use clear language"; Google Technical Writing "Words" |
| 12 | **Synonyms for one concept** | "library", "grid" and "model bank" used for the same thing in different sections. | Plain Language "Use the same terms consistently"; Google Technical Writing "Words" |
| 13 | **Exception before the rule** | "Except for spectra with S/N < 20, which ..., the fitter MUST ..." | Plain Language "Place the main idea before exceptions and conditions" |
| 14 | **Abstraction without examples (curse of knowledge)** | "Handles field geometry generally", with no worked case. | Google Technical Writing "Audience"; Heath and Heath (HBR 2006, summary only) |
| 15 | **Non-goals that nobody would expect anyway** | "Non-goal: a GUI for Windows 95." This wastes the section. | Google Technical Writing "Documents" (non-scope exercise) |
| 16 | **Generic or question headings** | "Introduction", "Miscellaneous", "What about tests?" | GOV.UK "Create a clear structure for your content" |
| 17 | **A reviewer who chases every finding** | Every review adds abstraction layers and tests for impossible cases. | Claude Code best practices, "Add an adversarial review step" |
| 18 | **Asking "is it clear?" instead of testing understanding** | A yes/no question to a colleague counts as reader testing. | Plain Language "Paraphrase testing" ("Avoid yes or no questions") |

---

## Which checks matter most for each part of the plan

| Plan part | Checks to weigh most |
|---|---|
| (1) Build specifications | A1-A11, B1-B5, B9, C2, C4, C6, C9, D1, D3 |
| (2) Verification loops | A1, A2, A3, A10, C1, C3, C7, D8, D9 |
| (3) Quality gates for pushing to `main` | A8, C1, C5, C8, C10, D9 (gate matrix as an if-then table), B6 (record gate decisions) |
| All three | B1, B2, B3, D1, D2, D4, D5, D6 |

---

## Sources

Opened on 2026-10-17 unless marked otherwise.

**Requirements**
- RFC 2119, Key words for use in RFCs to indicate requirement levels, section 6. https://www.rfc-editor.org/rfc/rfc2119
- RFC 8174, Ambiguity of uppercase vs lowercase in RFC 2119 key words. https://www.rfc-editor.org/rfc/rfc8174
- ISO/IEC/IEEE 29148:2018, *Systems and software engineering — Life cycle processes — Requirements engineering*. Not opened (paywalled). Characteristics taken from the secondary summary at https://www.modernrequirements.com/blogs/iso-29148-explained/ [check against the standard before citing it in a paper].
- Mavin A., Wilkinson P., Harwood A., Novak M. (2009), Easy Approach to Requirements Syntax (EARS), *17th IEEE Int. Requirements Engineering Conf.*, pp. 317-322, DOI 10.1109/RE.2009.9. Abstract opened at https://research.manchester.ac.uk/en/publications/easy-approach-to-requirements-syntax-ears/ ; patterns at https://alistairmavin.com/ears/ . The DOI comes from a search result [verify on the publisher's site].
- Femmer H., Méndez Fernández D., Wagner S., Eder S., Rapid quality assurance with Requirements Smells, *J. Syst. Softw.*, DOI 10.1016/j.jss.2016.02.047; arXiv:1611.08847 (abstract opened; the smell list comes from the PDF summary).
- Fowler M., Specification By Example (2004), https://martinfowler.com/bliki/SpecificationByExample.html ; GivenWhenThen (2013), https://martinfowler.com/bliki/GivenWhenThen.html
- Adzic G., *Specification by Example*, Manning 2011, https://gojko.net/books/specification-by-example/ (book page only; contents [unverified]).
- arc42, section 10 Quality requirements, https://docs.arc42.org/section-10/

**Document structure**
- Ubl M., Design Docs at Google (Industrial Empathy, 2020), https://www.industrialempathy.com/posts/design-docs-at-google/
- PEP 1, PEP Purpose and Guidelines, "What belongs in a successful PEP", https://peps.python.org/pep-0001/
- Rust RFC template, https://github.com/rust-lang/rfcs/blob/master/0000-template.md
- Nygard M., Documenting Architecture Decisions (2011), https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions
- Diátaxis, Start here, https://diataxis.fr/start-here/

**Agent-executable specs**
- GitHub Spec Kit: `spec-driven.md`, `templates/spec-template.md`, `templates/commands/checklist.md`, https://github.com/github/spec-kit
- Kiro specs, https://kiro.dev/docs/specs/ and https://kiro.dev/docs/specs/feature-specs/
- Anthropic, Best practices for Claude Code, https://code.claude.com/docs/en/best-practices (sections: Give Claude a way to verify its work; Provide specific context in your prompts; Write an effective CLAUDE.md; Let Claude interview you; Add an adversarial review step; Avoid common failure patterns)
- Nigh M., How to write a great agents.md: lessons from over 2,500 repositories, GitHub Blog, 2025-11-19, https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/
- Osmani A., How to write a good spec for AI agents, 2026-01-13, https://addyosmani.com/blog/good-spec/
- Wilson G. et al. (2014), Best Practices for Scientific Computing, *PLoS Biol* 12(1): e1001745, https://doi.org/10.1371/journal.pbio.1001745

**Readability**
- Federal Plain Language Guidelines (archived), https://github.com/GSA/plainlanguage.gov/tree/main/_pages/guidelines ; current guide series at https://digital.gov/guides/plain-language (page "Writing for understanding").
- GOV.UK, Writing to GOV.UK standards: "Create a clear structure for your content" and "Use clear language", https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/
- Google, Technical Writing One: "Audience", "Words", "Short sentences", "Documents", https://developers.google.com/tech-writing/one
- Heath C., Heath D., The Curse of Knowledge, *Harvard Business Review*, December 2006, https://hbr.org/2006/12/the-curse-of-knowledge (summary paragraph only; paywalled).
- BLUF / US Army AR 25-50: [unverified], not opened. The same principle is cited above from GOV.UK and Google.

**Limits of this review.** Four sources were not opened in full: the 29148 standard itself, the EARS paper body, Adzic's book and the HBR article. Claims that depend on them say so. Several sources about agents (Osmani, Nigh, Spec Kit, Kiro) are practitioner guidance or vendor documentation, not controlled studies. Treat them as current practice, not as evidence.
