# Progress Reporting

Class: `Rindow\RL\Agents\Util\ProgressBar`

`start($title,$numIterations,$maxDot)` initializes timing and prints an empty
bar. `update($iteration)` redraws elapsed and estimated remaining time.
`clearProgressBar()` temporarily removes it before a log line, and
`retrieveProgressBar()` redraws the previous state.

`laptime()` returns elapsed seconds; `laptimeString()` returns `MM:SS` or
`H:MM:SS`. Subclasses may override the protected `console()` method to redirect
output; the default destination is STDERR.
