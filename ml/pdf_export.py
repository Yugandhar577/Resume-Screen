"""Embedded-font PDF export with HarfBuzz shaping and faithful Unicode mappings."""
from io import BytesIO
from pathlib import Path
import re
from threading import Lock
import unicodedata

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfgen.textobject import bidiShapedText

_lock = Lock()
_fonts_ready = False
_font_root = Path(__file__).resolve().parent.parent/'public'/'fonts'


def _fonts():
    global _fonts_ready
    if not _fonts_ready:
        pdfmetrics.registerFont(TTFont('CareerSans', str(_font_root/'NotoSans-Regular.ttf'), shapable=True))
        pdfmetrics.registerFont(TTFont('CareerDeva', str(_font_root/'NotoSansDevanagari.ttf'), shapable=True))
        _fonts_ready = True


def _runs(text):
    runs = []
    current = 'CareerSans'
    for char in text:
        code = ord(char)
        if code < 32:
            raise ValueError(f'Unsupported control character U+{code:04X} in resume text.')
        if unicodedata.bidirectional(char) in ('R','AL','AN'):
            raise ValueError('Right-to-left script export is not supported by the current font and layout workflow.')
        if char in ('\u200c','\u200d'):
            font = current
        elif code in range(0x900,0x980) or code in range(0xA8E0,0xA900):
            font = 'CareerDeva'
        elif code < 128 and code >= 32:
            font = current
        else:
            font = 'CareerSans' if code in pdfmetrics.getFont('CareerSans').face.charToGlyph else 'CareerDeva'
        if char not in ('\u200c','\u200d') and code not in pdfmetrics.getFont(font).face.charToGlyph:
            raise ValueError(f'No bundled PDF font supports character U+{code:04X}. Use supported text or download the text copy.')
        if not runs or runs[-1][0] != font:
            runs.append((font,[char]))
        else:
            runs[-1][1].append(char)
        current = font
    return [(font,''.join(chars)) for font,chars in runs]


def _width(text):
    return sum(bidiShapedText(value, direction=None, fontName=font, fontSize=10, shaping=True)[1] for font,value in _runs(text))


def _clusters(text):
    clusters = []
    for char in text:
        if clusters and (unicodedata.category(char).startswith('M') or char in ('\u200c','\u200d') or clusters[-1].endswith(('\u094d','\u200c','\u200d'))):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _wrap(line, width):
    if not line:
        return ['']
    result, current = [], ''
    for token in re.findall(r'\S+|\s+', line):
        if current and _width(current+token) > width:
            result.append(current.rstrip()); current = ''
            if token.isspace():
                continue
        if _width(token) > width:
            for cluster in _clusters(token):
                if current and _width(current+cluster) > width:
                    result.append(current); current = ''
                current += cluster
        else:
            current += token
    if current:
        result.append(current.rstrip())
    return result


def create_pdf(text):
    normalized = text.replace('\r\n','\n').replace('\r','\n').replace('\t','    ')
    with _lock:
        _fonts()
        for line in normalized.split('\n'):
            _runs(line)  # Fail before writing if any glyph/script is unsupported.
        buffer = BytesIO()
        canvas = Canvas(buffer, pagesize=A4, pageCompression=1)
        canvas.setTitle('Resume'); canvas.setAuthor('')
        width, height = A4
        margin, leading = 51, 15
        y = height-margin
        for line in normalized.split('\n'):
            for fragment in _wrap(line,width-2*margin):
                if y < margin:
                    if canvas.getPageNumber() >= 30:
                        raise ValueError('PDF export is limited to 30 pages. Remove excessive blank lines or shorten the resume.')
                    canvas.showPage(); y = height-margin
                text_object = canvas.beginText(margin,y)
                for font, value in _runs(fragment):
                    text_object.setFont(font,10)
                    shaped, _ = bidiShapedText(value,direction=None,fontName=font,fontSize=10,shaping=True)
                    text_object.textOut(shaped)
                canvas.drawText(text_object)
                y -= leading
        canvas.save()
        return buffer.getvalue()
