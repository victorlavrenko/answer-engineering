// ae-rule-id: <dynamic>
// ae-stats: fired=3/3 (100.0%) total_applications=3 avg_repeat_when_fired=1.00
// ae-stats: chosen candidates: SSNHL 2, sudden sensorineural hearing loss 1
// ae-stats: top trigger terms: sudden 3/3
## Replace (once): sensorineural hearing loss

Prefix (any):

- sudden // ae-stats: matched=3/3

With:

- sudden sensorineural hearing loss // ae-stats: chosen=1/3 (33.3%) avg_hits_when_chosen=1.00 total_hits=1
- SSNHL // ae-stats: chosen=2/3 (66.7%) avg_hits_when_chosen=1.00 total_hits=2

## Rule activity summary
- most active rules by fired generations:
  - replace:sensorineural hearing loss: 3/3
- highest repeat burden:
  - replace:sensorineural hearing loss: 1.00
- fallback actually used:
  - none
// ae-stats: run-summary
//   applied_decisions=0 decision_limit_reached=false
