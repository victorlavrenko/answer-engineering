// ae-rule-id: <dynamic>
// ae-stats: fired=1/1 (100.0%) total_applications=2 avg_repeat_when_fired=2.00
// ae-stats: chosen candidates: keep original phrase 2
// ae-stats: top trigger terms: conductive 1/1, sensorineural 1/1
## Replace (once): hearing loss

Postfix: conductive | sensorineural // ae-stats: conductive 1/1, sensorineural 1/1

Fallback: keep original phrase // ae-stats: chosen=1/1 (100.0%) avg_hits_when_chosen=2.00 total_hits=2

## Rule activity summary
- most active rules by fired generations:
  - replace:hearing loss: 1/1
  - replace:hearing loss: 1/1
- highest repeat burden:
  - replace:hearing loss: 1.00
  - replace:hearing loss: 1.00
- fallback actually used:
  - replace:hearing loss: 1 fired generations
  - replace:hearing loss: 1 fired generations
// ae-stats: run-summary
//   applied_decisions=0 decision_limit_reached=false
