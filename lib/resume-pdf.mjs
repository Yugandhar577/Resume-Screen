import { jsPDF } from 'jspdf'

/** Generate a text PDF with an embedded font and line-by-line pagination. */
export function createResumePDF(text, fontBase64) {
  const doc = new jsPDF({ unit: 'mm', format: 'a4', putOnlyUsedFonts: true })
  doc.addFileToVFS('NotoSans-Regular.ttf', fontBase64)
  doc.addFont('NotoSans-Regular.ttf', 'NotoSans', 'normal')
  doc.setFont('NotoSans', 'normal')
  doc.setFontSize(10)
  const normalized = text.replace(/\r\n?/g, '\n').replace(/\t/g, '    ')
  const font = doc.getFont().metadata
  for (const char of normalized) {
    if (char !== '\n' && !font.characterToGlyph(char.codePointAt(0))) {
      throw new Error(`PDF font cannot render “${char}”. Replace that symbol before exporting; your resume text is unchanged.`)
    }
  }
  const margin = 18, bottom = 277, lineHeight = 5
  let y = margin
  for (const line of normalized.split('\n')) {
    const wrapped = line ? doc.splitTextToSize(line, 174) : ['']
    for (const fragment of wrapped) {
      if (y > bottom) { doc.addPage(); y = margin }
      if (fragment) doc.text(fragment, margin, y)
      y += lineHeight
    }
  }
  return doc
}
