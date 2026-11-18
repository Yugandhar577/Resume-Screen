"""Disposable document parser process; communicates through pipes, never files."""
import base64
import json
import os
import sys


def main():
    if os.name == 'nt':
        from ml.windows_limits import limit_current_process
        limit_current_process()
    else:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (40, 40))
        resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))
    from fastapi import HTTPException
    from ml.pdf_processing import extract_pdf
    try:
        raw = sys.stdin.buffer.read(8*1024*1024+1)
        if len(raw) > 8*1024*1024:
            raise HTTPException(status_code=413, detail='Document worker input is too large.')
        payload = json.loads(raw)
        result = extract_pdf(base64.b64decode(payload['content'], validate=True), payload['ocr'], payload['password'])
        if len(result['text']) > 50000:
            raise HTTPException(status_code=413, detail='Extracted resume text exceeds 50,000 characters.')
        output = {'result': result}
    except HTTPException as exc:
        output = {'error': str(exc.detail), 'status': exc.status_code}
    except Exception:
        output = {'error': 'Document processing failed. Upload a valid PDF or paste its text.', 'status': 400}
    sys.stdout.buffer.write(json.dumps(output, ensure_ascii=False).encode('utf-8'))


if __name__ == '__main__':
    main()
