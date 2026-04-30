// ae-rule-id: <dynamic>
// ae-stats: fired=1/1 (100.0%) total_applications=3 avg_repeat_when_fired=3.00
// ae-stats: chosen candidates: generated:probe_1 1, generated:probe_2 1, generated:probe_3 1
// ae-stats: top trigger terms: conductive 1/1, positive 1/1, Rinne 1/1
## Avoid (postfix, repeat): conductive

Prefix (all):

- Rinne // ae-stats: matched=1/1
- positive // ae-stats: matched=1/1

Postfix (all):

- conductive // ae-stats: matched=1/1

With:

- probe_1 // ae-stats: chosen=1/1 (100.0%) avg_hits_when_chosen=1.00 total_hits=1
- probe_2 // ae-stats: chosen=1/1 (100.0%) avg_hits_when_chosen=1.00 total_hits=1
- probe_3 // ae-stats: chosen=1/1 (100.0%) avg_hits_when_chosen=1.00 total_hits=1

## Rule activity summary
- most active rules by fired generations:
  - avoid:Rinne positive then conductive: 1/1
- highest repeat burden:
  - avoid:Rinne positive then conductive: 3.00
- fallback actually used:
  - none
// ae-stats: run-summary
//   applied_decisions=0 decision_limit_reached=false
