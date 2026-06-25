# Issue tracker: GitHub

Issues and PRDs for this repository live in GitHub issues for `ian-ross/ml-autoresearch`. Use the `gh` CLI for operations.

## Conventions

- Create: `gh issue create --repo ian-ross/ml-autoresearch --title "..." --body "..."`
- Read: `gh issue view --repo ian-ross/ml-autoresearch <number> --comments`
- List: `gh issue list --repo ian-ross/ml-autoresearch --state open --json number,title,body,labels,comments`
- Comment: `gh issue comment --repo ian-ross/ml-autoresearch <number> --body "..."`
- Apply/remove labels: `gh issue edit --repo ian-ross/ml-autoresearch <number> --add-label "..."` / `--remove-label "..."`
- Close: `gh issue close --repo ian-ross/ml-autoresearch <number> --comment "..."`

When run inside this clone, `gh` will usually infer the same repository from the git remote; the explicit `--repo` form avoids ambiguity.

## Skill wording

- "Publish to the issue tracker" means create a GitHub issue in `ian-ross/ml-autoresearch`.
- "Fetch the relevant ticket" means run `gh issue view --repo ian-ross/ml-autoresearch <number> --comments`.
