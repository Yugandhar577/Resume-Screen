import assert from 'node:assert/strict'
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { createResumePDF } from '../lib/resume-pdf.mjs'

const font = readFileSync('public/fonts/NotoSans-Regular.ttf').toString('base64')
const text = ['José García', 'jose@example.com', '', 'EXPERIENCE',
  ...Array.from({ length: 150 }, (_, i) => `- Entry ${i + 1}: Developed Python services with PostgreSQL and improved latency by 20%.`),
  'FINAL LINE: C++ C# — résumé café', 'Greek: Γειά σου', 'Cyrillic: Привет'].join('\n')
const doc = createResumePDF(text, font)
assert.ok(doc.getNumberOfPages() >= 3, 'Long resumes must paginate')
assert.throws(() => createResumePDF('Unsupported symbol 😀', font), /cannot render/)
mkdirSync('tmp/pdfs', { recursive: true })
writeFileSync('tmp/pdfs/export-regression.pdf', Buffer.from(doc.output('arraybuffer')))
console.log(`PDF export checks passed; generated ${doc.getNumberOfPages()} pages for text and visual QA.`)
