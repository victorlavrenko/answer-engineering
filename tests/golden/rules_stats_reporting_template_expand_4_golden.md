// ae-rule-id: <dynamic>
// ae-stats: fired=1/1 (100.0%) total_applications=4 avg_repeat_when_fired=4.00
// ae-stats: top trigger terms: conductive 1/1, forehead 1/1, left 1/1
## Avoid (last clause): contralateral conductive inference Weber

Scope: all

Prefix:

- Weber | forehead // ae-stats: Weber 1/1, forehead 1/1
- left || right // ae-stats: left 1/1, right 1/1

Postfix:

- right || left // ae-stats: right 1/1, left 1/1
- conductive // ae-stats: matched=1/1

## Rule activity summary
- most active rules by fired generations:
  - avoid:contralateral conductive inference Weber: 1/1
  - avoid:contralateral conductive inference Weber: 1/1
  - avoid:contralateral conductive inference Weber: 1/1
- highest repeat burden:
  - avoid:contralateral conductive inference Weber: 1.00
  - avoid:contralateral conductive inference Weber: 1.00
  - avoid:contralateral conductive inference Weber: 1.00
- fallback actually used:
  - none
// ae-stats: run-summary
//   applied_decisions=0 decision_limit_reached=false
