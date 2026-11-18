from io import BytesIO
import unittest

from fastapi.testclient import TestClient
from PyPDF2 import PdfReader

import main
from ml.pdf_export import create_pdf


class ExportTests(unittest.TestCase):
    def test_devanagari_combining_marks_and_mixed_scripts_roundtrip(self):
        text = 'किरण जोशी\nअनुभव\n- Python से 5 रिपोर्ट तैयार कीं।\nकौशल\nPython SQL\nJosé Γειά Привет\noffice efficiently'
        response = TestClient(main.app).post('/api/export-pdf',json={'resume':text})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers['content-type'],'application/pdf')
        result = PdfReader(BytesIO(response.content)).pages[0].extract_text().strip()
        self.assertEqual(result,text)
        self.assertEqual(response.headers['cache-control'],'no-store')

    def test_long_documents_paginate_without_losing_last_lines(self):
        text = '\n'.join([*[f'- Entry {i}: Built Python services for 50 users.' for i in range(1,151)],'FINAL LINE: résumé café'])
        reader = PdfReader(BytesIO(create_pdf(text)))
        result = '\n'.join(page.extract_text() for page in reader.pages)
        self.assertGreaterEqual(len(reader.pages),3)
        self.assertIn('FINAL LINE: résumé café',result)
        for i in range(1,151): self.assertIn(f'Entry {i}:',result)

    def test_markup_is_literal_and_unsupported_scripts_fail_visibly(self):
        text = '<script>alert(1)</script> & <img src="https://example.org">'
        self.assertEqual(PdfReader(BytesIO(create_pdf(text))).pages[0].extract_text().strip(),text)
        client = TestClient(main.app)
        for text in ('😀','العربية','\x00','   '):
            response = client.post('/api/export-pdf',json={'resume':text})
            self.assertEqual(response.status_code,422)
        with self.assertRaisesRegex(ValueError,'30 pages'):
            create_pdf('a\n'*2000)


if __name__ == '__main__':
    unittest.main()
