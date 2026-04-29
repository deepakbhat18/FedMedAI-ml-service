import io
import os
from datetime import datetime
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable


# ─── Color Palette ───────────────────────────────────────
DARK_BG      = colors.HexColor('#0d1117')
CARD_BG      = colors.HexColor('#161b22')
BORDER       = colors.HexColor('#30363d')
TEXT_PRIMARY = colors.HexColor('#e6edf3')
TEXT_GRAY    = colors.HexColor('#8b949e')
PURPLE       = colors.HexColor('#a855f7')
PINK         = colors.HexColor('#ff4d6d')
GREEN        = colors.HexColor('#00d68f')
AMBER        = colors.HexColor('#ffb224')
BLUE         = colors.HexColor('#3b7eff')
WHITE        = colors.white

RISK_COLORS = {
    'HIGH':     colors.HexColor('#ff4d6d'),
    'MODERATE': colors.HexColor('#ffb224'),
    'LOW':      colors.HexColor('#3b7eff'),
    'MINIMAL':  colors.HexColor('#00d68f'),
}

URGENCY_COLORS = {
    'IMMEDIATE':       colors.HexColor('#ff4d6d'),
    'WITHIN_48_HOURS': colors.HexColor('#ff6b35'),
    'WITHIN_1_WEEK':   colors.HexColor('#ffb224'),
    'WITHIN_1_MONTH':  colors.HexColor('#3b7eff'),
    'ROUTINE':         colors.HexColor('#00d68f'),
}


class MedicalPDFGenerator:
    """
    Generates professional medical diagnosis PDF reports.
    Dark theme, clean layout, includes:
    - Patient information
    - Disease predictions with risk levels
    - AI recommendations
    - Doctor signature section
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Define custom paragraph styles."""
        self.style_title = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Normal'],
            fontSize=22,
            textColor=WHITE,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER,
            spaceAfter=4,
        )
        self.style_subtitle = ParagraphStyle(
            'CustomSubtitle',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=TEXT_GRAY,
            fontName='Helvetica',
            alignment=TA_CENTER,
            spaceAfter=2,
        )
        self.style_section = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=PURPLE,
            fontName='Helvetica-Bold',
            spaceBefore=12,
            spaceAfter=6,
        )
        self.style_body = ParagraphStyle(
            'Body',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=TEXT_PRIMARY,
            fontName='Helvetica',
            leading=14,
        )
        self.style_small = ParagraphStyle(
            'Small',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=TEXT_GRAY,
            fontName='Helvetica',
            leading=12,
        )
        self.style_bold = ParagraphStyle(
            'Bold',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=TEXT_PRIMARY,
            fontName='Helvetica-Bold',
        )
        self.style_disclaimer = ParagraphStyle(
            'Disclaimer',
            parent=self.styles['Normal'],
            fontSize=7.5,
            textColor=TEXT_GRAY,
            fontName='Helvetica-Oblique',
            alignment=TA_CENTER,
            leading=11,
        )

    def generate(
        self,
        patient_data: Dict,
        predictions: List[Dict],
        recommendations: Optional[Dict] = None,
        hospital_data: Optional[Dict] = None,
        features: Optional[Dict] = None,
    ) -> bytes:
        """
        Generate complete PDF report.

        Returns:
            PDF as bytes
        """
        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=18*mm,
            leftMargin=18*mm,
            topMargin=16*mm,
            bottomMargin=16*mm,
            title=f"Medical Report - {patient_data.get('patientCode', 'Unknown')}",
        )

        story = []

        # Build sections
        story += self._build_header(hospital_data, patient_data)
        story += self._build_patient_info(patient_data, features)
        story += self._build_predictions(predictions)

        if recommendations and recommendations.get('disease_recommendations'):
            story += self._build_recommendations(recommendations)

        story += self._build_vitals_table(features)
        story += self._build_footer(patient_data, hospital_data)

        # Build PDF with dark background
        doc.build(
            story,
            onFirstPage=self._draw_background,
            onLaterPages=self._draw_background,
        )

        return buffer.getvalue()

    # ─── Background ─────────────────────────────────────────
    def _draw_background(self, canvas, doc):
        """Draw dark background on every page."""
        canvas.saveState()
        canvas.setFillColor(DARK_BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    # ─── Header ─────────────────────────────────────────────
    def _build_header(self, hospital: Optional[Dict], patient: Dict) -> list:
        story = []

        hosp_name = hospital.get('name', 'FL Medical Center') if hospital else 'FL Medical Center'

        # Top gradient bar (simulated with colored table)
        bar_data = [['']]
        bar_table = Table(bar_data, colWidths=[174*mm], rowHeights=[4])
        bar_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PURPLE),
            ('LINEBELOW', (0, 0), (-1, -1), 2, PINK),
        ]))
        story.append(bar_table)
        story.append(Spacer(1, 8*mm))

        # Hospital name
        story.append(Paragraph(f'🏥 {hosp_name}', self.style_title))
        story.append(Paragraph(
            'Federated Learning · AI Medical Diagnosis Report',
            self.style_subtitle
        ))
        story.append(Spacer(1, 2*mm))

        # Report metadata bar
        report_date = datetime.now().strftime('%B %d, %Y  %H:%M')
        report_id   = f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        meta_data = [[
            Paragraph(f'<font color="#8b949e">Report ID:</font> {report_id}', self.style_small),
            Paragraph(f'<font color="#8b949e">Generated:</font> {report_date}', self.style_small),
            Paragraph('<font color="#a855f7">CONFIDENTIAL</font>', ParagraphStyle(
                'conf', parent=self.styles['Normal'],
                fontSize=8, textColor=PURPLE,
                fontName='Helvetica-Bold', alignment=TA_RIGHT,
            )),
        ]]
        meta_table = Table(meta_data, colWidths=[60*mm, 70*mm, 44*mm])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND',  (0, 0), (-1, -1), CARD_BG),
            ('ROUNDEDCORNERS', (0, 0), (-1, -1), 4),
            ('TOPPADDING',  (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('BOX', (0, 0), (-1, -1), 0.5, BORDER),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 6*mm))
        return story

    # ─── Patient Info ────────────────────────────────────────
    def _build_patient_info(self, patient: Dict, features: Optional[Dict]) -> list:
        story = []
        story.append(Paragraph('PATIENT INFORMATION', self.style_section))

        age    = patient.get('age', 'N/A')
        gender = patient.get('gender', 'N/A')
        hosp   = patient.get('hospitalName', 'N/A')
        code   = patient.get('patientCode', 'N/A')
        diag   = patient.get('diagnosis', 'Pending AI Analysis')

        rows = [
            [
                Paragraph('<font color="#8b949e">Patient Code</font>', self.style_small),
                Paragraph(str(code), self.style_bold),
                Paragraph('<font color="#8b949e">Age</font>', self.style_small),
                Paragraph(str(age), self.style_bold),
            ],
            [
                Paragraph('<font color="#8b949e">Gender</font>', self.style_small),
                Paragraph(str(gender), self.style_bold),
                Paragraph('<font color="#8b949e">Hospital</font>', self.style_small),
                Paragraph(str(hosp), self.style_bold),
            ],
            [
                Paragraph('<font color="#8b949e">Diagnosis</font>', self.style_small),
                Paragraph(str(diag), self.style_bold),
                Paragraph('<font color="#8b949e">Report Type</font>', self.style_small),
                Paragraph('Multi-Disease AI Screening', self.style_bold),
            ],
        ]

        t = Table(rows, colWidths=[35*mm, 55*mm, 35*mm, 49*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0, 0), (-1, -1), CARD_BG),
            ('BOX',           (0, 0), (-1, -1), 0.5, BORDER),
            ('INNERGRID',     (0, 0), (-1, -1), 0.3, BORDER),
            ('TOPPADDING',    (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ]))
        story.append(t)
        story.append(Spacer(1, 5*mm))
        return story

    # ─── Predictions ─────────────────────────────────────────
    def _build_predictions(self, predictions: List[Dict]) -> list:
        story = []
        story.append(Paragraph('AI DISEASE RISK ASSESSMENT', self.style_section))

        disease_emojis = {
            'heart_disease':  '❤',
            'diabetes':       '♦',
            'kidney_disease': '◆',
            'liver_disease':  '▲',
            'stroke':         '●',
        }

        # Header row
        header = [
            Paragraph('DISEASE', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=8, textColor=TEXT_GRAY, fontName='Helvetica-Bold')),
            Paragraph('RISK LEVEL', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=8, textColor=TEXT_GRAY, fontName='Helvetica-Bold', alignment=TA_CENTER)),
            Paragraph('CONFIDENCE', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=8, textColor=TEXT_GRAY, fontName='Helvetica-Bold', alignment=TA_CENTER)),
            Paragraph('RESULT', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=8, textColor=TEXT_GRAY, fontName='Helvetica-Bold', alignment=TA_CENTER)),
            Paragraph('PROBABILITY', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=8, textColor=TEXT_GRAY, fontName='Helvetica-Bold')),
        ]
        rows = [header]

        for pred in sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True):
            risk      = pred.get('risk_level', 'MINIMAL')
            risk_col  = RISK_COLORS.get(risk, GREEN)
            conf      = pred.get('confidence', 0)
            conf_pct  = f"{conf * 100:.1f}%"
            disease   = pred.get('disease_name', pred.get('disease_type', 'Unknown'))
            emoji     = disease_emojis.get(pred.get('disease_type', ''), '•')
            label     = pred.get('prediction_label', 'Unknown')
            healthy   = f"{(1 - conf) * 100:.1f}%"

            # Progress bar (text-based)
            bar_filled = int(conf * 20)
            bar = '█' * bar_filled + '░' * (20 - bar_filled)

            rows.append([
                Paragraph(f'{emoji}  {disease}', self.style_body),
                Paragraph(
                    f'<font color="#{risk_col.hexval()[2:]}">{risk}</font>',
                    ParagraphStyle('rc', parent=self.styles['Normal'],
                        fontSize=9, fontName='Helvetica-Bold', alignment=TA_CENTER,
                        textColor=risk_col)
                ),
                Paragraph(conf_pct, ParagraphStyle('cc', parent=self.styles['Normal'],
                    fontSize=10, fontName='Helvetica-Bold', alignment=TA_CENTER,
                    textColor=risk_col)),
                Paragraph(label, ParagraphStyle('lc', parent=self.styles['Normal'],
                    fontSize=8, fontName='Helvetica', alignment=TA_CENTER,
                    textColor=TEXT_PRIMARY)),
                Paragraph(
                    f'<font color="#{risk_col.hexval()[2:]}">{bar}</font>  '
                    f'<font color="#8b949e">H:{healthy}</font>',
                    ParagraphStyle('bc', parent=self.styles['Normal'],
                        fontSize=7, fontName='Courier', textColor=risk_col)
                ),
            ])

        t = Table(rows, colWidths=[47*mm, 28*mm, 24*mm, 38*mm, 37*mm])
        t.setStyle(TableStyle([
            # Header
            ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor('#1c2128')),
            ('TOPPADDING',    (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            # Body rows
            ('BACKGROUND',    (0, 1), (-1, -1), CARD_BG),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [CARD_BG, colors.HexColor('#1a1f2a')]),
            ('TOPPADDING',    (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 9),
            # All
            ('LEFTPADDING',   (0, 0), (-1, -1), 10),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
            ('BOX',           (0, 0), (-1, -1), 0.5, BORDER),
            ('INNERGRID',     (0, 0), (-1, -1), 0.3, BORDER),
            ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 5*mm))
        return story

    # ─── Recommendations ─────────────────────────────────────
    def _build_recommendations(self, recommendations: Dict) -> list:
        story = []
        story.append(Paragraph('MEDICAL RECOMMENDATIONS', self.style_section))

        # Overall summary
        summary = recommendations.get('overall_summary', '')
        if summary:
            sum_data = [[Paragraph(summary, self.style_body)]]
            sum_table = Table(sum_data, colWidths=[174*mm])
            sum_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#1c2128')),
                ('BOX',        (0, 0), (-1, -1), 0.5, BLUE),
                ('LEFTPADDING',  (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING',   (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING',(0, 0), (-1, -1), 8),
            ]))
            story.append(sum_table)
            story.append(Spacer(1, 4*mm))

        # Priority + Follow-up
        priority = recommendations.get('most_urgent', '')
        followup = recommendations.get('follow_up_timeline', '')
        if priority or followup:
            pf_data = [[
                Paragraph(
                    f'<font color="#ffb224">⚡ Priority Action:</font> {priority}',
                    self.style_small
                ),
                Paragraph(
                    f'<font color="#8b949e">📅 Follow-up:</font> {followup}',
                    self.style_small
                ),
            ]]
            pf_table = Table(pf_data, colWidths=[87*mm, 87*mm])
            pf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
                ('BOX',        (0, 0), (-1, -1), 0.5, BORDER),
                ('LEFTPADDING',  (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING',   (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING',(0, 0), (-1, -1), 7),
            ]))
            story.append(pf_table)
            story.append(Spacer(1, 4*mm))

        # Per-disease recommendations
        for rec in recommendations.get('disease_recommendations', []):
            story.append(KeepTogether(
                self._build_single_recommendation(rec)
            ))
            story.append(Spacer(1, 3*mm))

        # General advice
        advice = recommendations.get('general_advice', [])
        if advice:
            story.append(Paragraph('General Health Advice', ParagraphStyle(
                'ga', parent=self.styles['Normal'],
                fontSize=9, textColor=GREEN,
                fontName='Helvetica-Bold', spaceBefore=4, spaceAfter=4,
            )))
            advice_rows = [[
                Paragraph(f'✓  {a}', self.style_small)
                for a in advice[i:i+2]
            ] for i in range(0, len(advice), 2)]

            # Pad last row if odd
            if advice_rows and len(advice_rows[-1]) == 1:
                advice_rows[-1].append(Paragraph('', self.style_small))

            if advice_rows:
                adv_table = Table(advice_rows, colWidths=[87*mm, 87*mm])
                adv_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
                    ('BOX',        (0, 0), (-1, -1), 0.5, BORDER),
                    ('INNERGRID',  (0, 0), (-1, -1), 0.3, BORDER),
                    ('LEFTPADDING',  (0, 0), (-1, -1), 10),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                    ('TOPPADDING',   (0, 0), (-1, -1), 5),
                    ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
                ]))
                story.append(adv_table)

        story.append(Spacer(1, 5*mm))
        return story

    def _build_single_recommendation(self, rec: Dict) -> list:
        """Build one disease recommendation block."""
        items = []
        risk      = rec.get('risk_level', 'MODERATE')
        risk_col  = RISK_COLORS.get(risk, AMBER)
        urgency   = rec.get('urgency', 'ROUTINE')
        urg_col   = URGENCY_COLORS.get(urgency, BLUE)
        disease   = rec.get('disease_name', 'Unknown')
        conf      = rec.get('confidence', 0)
        urgency_label = urgency.replace('_', ' ')

        # Disease header row
        header_data = [[
            Paragraph(
                f'<font color="#{risk_col.hexval()[2:]}">{disease}</font>',
                ParagraphStyle('dh', parent=self.styles['Normal'],
                    fontSize=10, fontName='Helvetica-Bold', textColor=risk_col)
            ),
            Paragraph(
                f'<font color="#{risk_col.hexval()[2:]}">{risk} · {conf:.1f}%</font>',
                ParagraphStyle('ri', parent=self.styles['Normal'],
                    fontSize=8, fontName='Helvetica-Bold',
                    textColor=risk_col, alignment=TA_RIGHT)
            ),
            Paragraph(
                f'<font color="#{urg_col.hexval()[2:]}">{urgency_label}</font>',
                ParagraphStyle('ui', parent=self.styles['Normal'],
                    fontSize=8, fontName='Helvetica-Bold',
                    textColor=urg_col, alignment=TA_RIGHT)
            ),
        ]]
        header_t = Table(header_data, colWidths=[80*mm, 47*mm, 47*mm])
        header_t.setStyle(TableStyle([
            ('BACKGROUND',   (0, 0), (-1, -1), colors.HexColor('#1c2128')),
            ('BOX',          (0, 0), (-1, -1), 0.5, risk_col),
            ('LEFTPADDING',  (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING',   (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 7),
            ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        items.append(header_t)

        # Explanation
        explanation = rec.get('explanation', '')
        if explanation:
            exp_data = [[Paragraph(f'💡  {explanation}', self.style_small)]]
            exp_t = Table(exp_data, colWidths=[174*mm])
            exp_t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
                ('LEFTPADDING',  (0, 0), (-1, -1), 12),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING',   (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
            ]))
            items.append(exp_t)

        # Tests / Specialists / Lifestyle - 3 columns
        tests     = rec.get('tests_recommended', [])[:5]
        specs     = rec.get('specialists_to_consult', [])
        lifestyle = rec.get('lifestyle_changes', [])[:4]

        def make_list(title, color, items_list):
            if not items_list:
                return Paragraph('', self.style_small)
            lines = [Paragraph(
                f'<font color="#{color.hexval()[2:]}">{title}</font>',
                ParagraphStyle('lt', parent=self.styles['Normal'],
                    fontSize=7.5, fontName='Helvetica-Bold', textColor=color,
                    spaceAfter=2)
            )]
            for item in items_list:
                lines.append(Paragraph(f'→  {item}', self.style_small))
            return lines

        col1 = make_list('🔬 TESTS', BLUE, tests)
        col2 = make_list('👨‍⚕️ SPECIALISTS', PURPLE, specs) + \
               [Spacer(1, 3)] + \
               make_list('🥗 LIFESTYLE', GREEN, lifestyle)
        col3_med = rec.get('medications_note', '')
        col3 = (
            [Paragraph('<font color="#ffb224">💊 MEDICATIONS</font>',
                ParagraphStyle('mt', parent=self.styles['Normal'],
                    fontSize=7.5, fontName='Helvetica-Bold',
                    textColor=AMBER, spaceAfter=2)),
             Paragraph(col3_med, self.style_small)]
            if col3_med else [Paragraph('', self.style_small)]
        )

        body_data = [[col1, col2, col3]]
        body_t = Table(body_data, colWidths=[58*mm, 70*mm, 46*mm])
        body_t.setStyle(TableStyle([
            ('BACKGROUND',   (0, 0), (-1, -1), CARD_BG),
            ('BOX',          (0, 0), (-1, -1), 0.5, BORDER),
            ('INNERGRID',    (0, 0), (-1, -1), 0.3, BORDER),
            ('LEFTPADDING',  (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING',   (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 8),
            ('VALIGN',       (0, 0), (-1, -1), 'TOP'),
        ]))
        items.append(body_t)
        return items

    # ─── Vitals Table ────────────────────────────────────────
    def _build_vitals_table(self, features: Optional[Dict]) -> list:
        if not features:
            return []
        story = []
        story.append(Paragraph('PATIENT VITALS USED FOR ANALYSIS', self.style_section))

        FEATURE_MAP = {
            'age':           ('Age',            'years', 'N/A'),
            'bloodPressure': ('Blood Pressure',  'mmHg',  '<120'),
            'glucose':       ('Blood Glucose',   'mg/dl', '70-100'),
            'cholesterol':   ('Cholesterol',     'mg/dl', '<200'),
            'bmi':           ('BMI',             'kg/m²', '18.5-24.9'),
            'heartRate':     ('Heart Rate',      'bpm',   '60-100'),
            'creatinine':    ('Creatinine',      'mg/dl', '0.6-1.2'),
            'hemoglobin':    ('Hemoglobin',      'g/dl',  '12-17.5'),
            'bilirubin':     ('Bilirubin',       'mg/dl', '0.1-1.2'),
            'altEnzyme':     ('ALT Enzyme',      'U/L',   '<40'),
            'sodium':        ('Sodium',          'mEq/L', '136-145'),
            'potassium':     ('Potassium',       'mEq/L', '3.5-5.0'),
        }

        THRESHOLDS = {
            'bloodPressure': 130, 'glucose': 100, 'cholesterol': 200,
            'bmi': 25, 'creatinine': 1.2, 'bilirubin': 1.2, 'altEnzyme': 40,
        }

        rows = [[
            Paragraph('PARAMETER', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=7.5, textColor=TEXT_GRAY, fontName='Helvetica-Bold')),
            Paragraph('VALUE', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=7.5, textColor=TEXT_GRAY, fontName='Helvetica-Bold', alignment=TA_CENTER)),
            Paragraph('UNIT', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=7.5, textColor=TEXT_GRAY, fontName='Helvetica-Bold', alignment=TA_CENTER)),
            Paragraph('NORMAL RANGE', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=7.5, textColor=TEXT_GRAY, fontName='Helvetica-Bold', alignment=TA_CENTER)),
            Paragraph('STATUS', ParagraphStyle('h', parent=self.styles['Normal'],
                fontSize=7.5, textColor=TEXT_GRAY, fontName='Helvetica-Bold', alignment=TA_CENTER)),
        ]]

        for key, (name, unit, normal) in FEATURE_MAP.items():
            val = features.get(key)
            if val is None:
                continue
            try:
                val_f = float(val)
            except (ValueError, TypeError):
                continue

            threshold = THRESHOLDS.get(key)
            if threshold and val_f > threshold:
                status_text = '⚠ HIGH'
                status_col  = PINK
                val_col     = PINK
            else:
                status_text = '✓ Normal'
                status_col  = GREEN
                val_col     = TEXT_PRIMARY

            rows.append([
                Paragraph(name, self.style_small),
                Paragraph(
                    f'<font color="#{val_col.hexval()[2:]}">{val_f}</font>',
                    ParagraphStyle('vc', parent=self.styles['Normal'],
                        fontSize=9, fontName='Helvetica-Bold',
                        textColor=val_col, alignment=TA_CENTER)
                ),
                Paragraph(unit,   ParagraphStyle('uc', parent=self.styles['Normal'],
                    fontSize=8, textColor=TEXT_GRAY, alignment=TA_CENTER)),
                Paragraph(normal, self.style_small),
                Paragraph(
                    f'<font color="#{status_col.hexval()[2:]}">{status_text}</font>',
                    ParagraphStyle('sc', parent=self.styles['Normal'],
                        fontSize=8, fontName='Helvetica-Bold',
                        textColor=status_col, alignment=TA_CENTER)
                ),
            ])

        # Boolean fields
        for key, label in [
            ('smoking', 'Smoking'), ('diabetesHistory', 'Diabetes History'),
            ('familyHistory', 'Family History'),
        ]:
            val = features.get(key)
            if val is not None:
                is_yes = str(val) in ('1', '1.0', 'True', 'true')
                rows.append([
                    Paragraph(label, self.style_small),
                    Paragraph(
                        f'<font color="{"#ff4d6d" if is_yes else "#00d68f"}">'
                        f'{"Yes" if is_yes else "No"}</font>',
                        ParagraphStyle('bc', parent=self.styles['Normal'],
                            fontSize=9, fontName='Helvetica-Bold',
                            textColor=PINK if is_yes else GREEN, alignment=TA_CENTER)
                    ),
                    Paragraph('—', self.style_small),
                    Paragraph('—', self.style_small),
                    Paragraph(
                        f'<font color="{"#ff4d6d" if is_yes else "#00d68f"}">'
                        f'{"⚠ Risk Factor" if is_yes else "✓ None"}</font>',
                        ParagraphStyle('bs', parent=self.styles['Normal'],
                            fontSize=8, fontName='Helvetica-Bold',
                            textColor=PINK if is_yes else GREEN, alignment=TA_CENTER)
                    ),
                ])

        t = Table(rows, colWidths=[52*mm, 28*mm, 20*mm, 38*mm, 36*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1c2128')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [CARD_BG, colors.HexColor('#1a1f2a')]),
            ('BOX',        (0, 0), (-1, -1), 0.5, BORDER),
            ('INNERGRID',  (0, 0), (-1, -1), 0.3, BORDER),
            ('LEFTPADDING',  (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING',   (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
            ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 5*mm))
        return story

    # ─── Footer ─────────────────────────────────────────────
    def _build_footer(self, patient: Dict, hospital: Optional[Dict]) -> list:
        story = []

        # Doctor signature area
        sig_data = [[
            Paragraph('Attending Doctor', ParagraphStyle('sl', parent=self.styles['Normal'],
                fontSize=8, textColor=TEXT_GRAY, fontName='Helvetica')),
            Paragraph('Doctor Signature', ParagraphStyle('sl', parent=self.styles['Normal'],
                fontSize=8, textColor=TEXT_GRAY, fontName='Helvetica', alignment=TA_CENTER)),
            Paragraph('Date', ParagraphStyle('sl', parent=self.styles['Normal'],
                fontSize=8, textColor=TEXT_GRAY, fontName='Helvetica', alignment=TA_RIGHT)),
        ], [
            Paragraph('_' * 30, self.style_small),
            Paragraph('_' * 30, ParagraphStyle('ss', parent=self.styles['Normal'],
                fontSize=9, textColor=TEXT_GRAY, alignment=TA_CENTER)),
            Paragraph(datetime.now().strftime('%d / %m / %Y'),
                ParagraphStyle('ds', parent=self.styles['Normal'],
                    fontSize=9, textColor=TEXT_PRIMARY,
                    fontName='Helvetica-Bold', alignment=TA_RIGHT)),
        ]]
        sig_t = Table(sig_data, colWidths=[60*mm, 70*mm, 44*mm])
        sig_t.setStyle(TableStyle([
            ('BACKGROUND',   (0, 0), (-1, -1), CARD_BG),
            ('BOX',          (0, 0), (-1, -1), 0.5, BORDER),
            ('LEFTPADDING',  (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING',   (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 12),
        ]))
        story.append(sig_t)
        story.append(Spacer(1, 5*mm))

        # Bottom bar
        bar_data = [['']]
        bar_t = Table(bar_data, colWidths=[174*mm], rowHeights=[2])
        bar_t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PURPLE),
        ]))
        story.append(bar_t)
        story.append(Spacer(1, 3*mm))

        # Disclaimer
        story.append(Paragraph(
            '⚕️  This report is generated by an AI-powered Federated Learning system and is intended '
            'to assist medical professionals. It does NOT replace professional medical advice, '
            'diagnosis, or treatment. Always consult a qualified healthcare provider.',
            self.style_disclaimer
        ))
        story.append(Paragraph(
            f'Powered by Federated Learning Medical AI  ·  '
            f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}  ·  CONFIDENTIAL',
            self.style_disclaimer
        ))
        return story


# Global instance
pdf_generator = MedicalPDFGenerator()