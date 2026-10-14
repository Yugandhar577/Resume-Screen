"""Local text/column extraction and optional bundled OCR; no provider uploads."""
import io
import importlib.util
from threading import Lock
from fastapi import HTTPException

_ocr = None
_ocr_lock = Lock()


def ocr_available():
    return importlib.util.find_spec('rapidocr_onnxruntime') is not None and importlib.util.find_spec('pypdfium2') is not None


def column_text(page):
    words = page.extract_words()
    if not words:
        return '', 'no_text'
    rows = []
    for word in sorted(words, key=lambda w: (w['top'], w['x0'])):
        if not rows or abs(word['top'] - rows[-1][0]['top']) > 3:
            rows.append([word])
        else:
            rows[-1].append(word)
    middle = page.width / 2
    dual = [row for row in rows if any(w['x1'] < middle-15 for w in row) and any(w['x0'] > middle+15 for w in row)]
    if len(dual) >= 4:
        first = min(row[0]['top'] for row in dual)
        body = [w for w in words if w['top'] >= first]
        crossings = [w for w in body if w['x0'] < middle < w['x1']]
        if not crossings:
            header = ' '.join(w['text'] for row in rows if row[0]['top'] < first for w in sorted(row, key=lambda w:w['x0']))
            columns = []
            for side in (False, True):
                selected = [[w for w in row if (w['x0'] > middle) == side] for row in rows if row[0]['top'] >= first]
                columns.append('\n'.join(' '.join(w['text'] for w in sorted(row, key=lambda w:w['x0'])) for row in selected if row))
            return '\n'.join([header, *columns]).strip(), 'two_column_heuristic'
    return page.extract_text(x_tolerance=2, y_tolerance=3) or '', 'pdf_text'


def local_ocr(content, page_indices):
    if not ocr_available():
        raise HTTPException(status_code=503, detail='Local OCR is not installed. Install requirements-ocr.txt or paste extracted text.')
    if len(page_indices) > 5:
        raise HTTPException(status_code=413, detail='OCR is limited to five scanned pages per request.')
    import numpy as np
    import pypdfium2
    from rapidocr_onnxruntime import RapidOCR
    global _ocr
    results = {}
    with _ocr_lock:
        if _ocr is None:
            _ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        document = pypdfium2.PdfDocument(content)
        try:
            for index in page_indices:
                page = document[index]
                try:
                    width, height = page.get_size()
                    if width * height > 2000000:
                        raise HTTPException(status_code=413, detail='PDF page dimensions are too large for OCR.')
                    bitmap = page.render(scale=1.5)
                    try:
                        image = np.array(bitmap.to_pil().convert('RGB'))[:, :, ::-1].copy()
                        detected, _ = _ocr(image)
                        results[index] = '\n'.join(item[1] for item in (detected or []) if float(item[2]) >= 0.5)
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        finally:
            document.close()
    return results


def extract_pdf(content, use_ocr=False, password=''):
    from PyPDF2 import PdfReader, PdfWriter
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            if not reader.decrypt(password):
                raise HTTPException(status_code=400, detail='This PDF needs the correct password. Enter it locally or upload an unlocked copy.')
            writer = PdfWriter(); writer.append_pages_from_reader(reader)
            buffer = io.BytesIO(); writer.write(buffer); content = buffer.getvalue()
        if len(reader.pages) > 30:
            raise HTTPException(status_code=413, detail='PDF resumes must contain at most 30 pages.')
    except HTTPException:
        raise
    except Exception:
        reader = None  # pdfplumber may be able to parse a file the first parser cannot.
    pages, methods, warnings = [], [], []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            if len(pdf.pages) > 30:
                raise HTTPException(status_code=413, detail='PDF resumes must contain at most 30 pages.')
            for page in pdf.pages:
                text, method = column_text(page)
                pages.append(text); methods.append(method)
    except HTTPException:
        raise
    except Exception:
        if reader is None:
            raise HTTPException(status_code=400, detail='Could not read this PDF. Upload a valid PDF or paste the text.')
        pages = [page.extract_text() or '' for page in reader.pages]; methods = ['primary_parser'] * len(pages)
    scanned = [index for index, text in enumerate(pages) if not text.strip()]
    if scanned:
        if not use_ocr:
            if not any(text.strip() for text in pages):
                raise HTTPException(status_code=400, detail='No readable text found. Enable local OCR for scanned PDFs or paste the text.')
            warnings.append('Some pages have no readable text. Enable OCR if they contain scanned resume content.')
        else:
            for index, text in local_ocr(content, scanned).items():
                pages[index] = text; methods[index] = 'local_ocr'
            warnings.append('OCR text may contain recognition errors. Review the extracted resume before accepting edits.')
    if 'two_column_heuristic' in methods:
        warnings.append('Two-column reading order was inferred. Review extracted text for section order and completeness.')
    text = '\n\n'.join(pages).strip()
    if not text:
        raise HTTPException(status_code=400, detail='No readable resume text was recovered.')
    return {'text': text, 'methods': methods, 'warnings': warnings, 'pages': len(pages)}
