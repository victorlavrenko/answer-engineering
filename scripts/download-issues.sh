gh issue list \
  --repo victorlavrenko/answer-engineering \
  --state all \
  --limit 1000 \
  --json number,title,body \
  --template '
{{range .}}
# Issue #{{.number}}: {{.title}}

{{.body}}

---
{{end}}
'