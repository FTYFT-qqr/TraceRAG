"""定义阅读工作台的视觉变量及原生控件样式。"""

WORKSPACE_CSS = """
<style>
:root {
 --paper-desk:#f5f5f0; --paper-sheet:#fff; --paper-inset:#f0f2ed;
 --archive-ink:#20352f; --archive-secondary:#53645e; --archive-muted:#6b7770;
 --archive-rule:#dce3db; --evidence-green:#235b46; --evidence-wash:#eaf2eb;
 --absence-amber:#875318; --delete-red:#9e3f36;
}
html,body,[data-testid="stAppViewContainer"] {background:var(--paper-desk);color:var(--archive-ink);font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;-webkit-font-smoothing:antialiased;}
[data-testid="stMainBlockContainer"] {max-width:1280px;padding:4rem 3rem 4rem;}
[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stHeader"] {background:var(--paper-desk);}
[data-testid="stSidebar"] {background:var(--paper-desk);border-right:1px solid var(--archive-rule);}
[data-testid="stSidebarContent"] {padding-top:1.5rem;}
h1,h2,h3,p {color:var(--archive-ink);}
h1 {font-size:2rem!important;font-weight:650!important;letter-spacing:-.04em;}
h2 {font-size:1.35rem!important;font-weight:650!important;}
h3 {font-size:1.1rem!important;font-weight:600!important;}
p,[data-testid="stMarkdownContainer"] li {line-height:1.75;}
[data-testid="stCaptionContainer"] {color:var(--archive-muted);font-size:.875rem;}
[data-testid="stVerticalBlockBorderWrapper"]>div {background:var(--paper-sheet);border-color:var(--archive-rule)!important;border-radius:14px!important;}
.st-key-upload_panel,.st-key-empty_library,.st-key-answer_panel,.st-key-question_panel,
[class*="st-key-document_"],[class*="st-key-source_"] {background:var(--paper-sheet);border-color:var(--archive-rule)!important;border-radius:14px;}
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea {background:var(--paper-inset);color:var(--archive-ink);font-size:1rem;}
[data-testid="stTextArea"] textarea {min-height:130px;line-height:1.75;}
[data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button {min-height:44px;border-radius:8px;border-color:var(--archive-rule);font-weight:500;transition:background-color 160ms ease,transform 120ms ease;}
button[kind="primary"],button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primaryFormSubmit"] {background:var(--evidence-green)!important;border-color:var(--evidence-green)!important;color:white!important;}
button[kind="primary"] p,button[data-testid="stBaseButton-primaryFormSubmit"] p {color:white!important;}
button:focus-visible,input:focus-visible,textarea:focus-visible {outline:3px solid var(--evidence-green)!important;outline-offset:3px;}
button:active {transform:scale(.98);}
button:disabled {opacity:.5;cursor:not-allowed;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:8px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:12px 16px;border-radius:8px;min-height:44px;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:var(--evidence-wash);}
[data-testid="stRadioOption"]:has(input:checked)>div>div:first-child {background:var(--evidence-green);border-color:var(--evidence-green);}
[data-testid="stFileUploaderDropzone"] {background:var(--paper-inset);border:1px dashed var(--archive-rule);border-radius:12px;}
[data-testid="stFileUploaderDropzone"] small {color:var(--archive-secondary);}
[data-testid="stExpander"] {border-color:var(--archive-rule);border-radius:10px;}
.tr-brand {display:flex;align-items:center;gap:12px;margin:0 0 28px;}
.tr-mark {width:36px;height:36px;display:grid;place-items:center;background:var(--evidence-green);border-radius:10px;}
.tr-brand-name {font-size:20px;font-weight:700;letter-spacing:-.5px;}
.tr-brand-caption {color:var(--archive-muted);font-size:12px;margin-top:2px;}
.tr-eyebrow {font-size:12px;letter-spacing:.12em;color:var(--evidence-green);font-weight:650;margin-bottom:8px;}
.tr-heading {font-family:"STSong","SimSun",serif;font-size:32px;font-weight:600;letter-spacing:-.5px;margin:0 0 8px;}
.tr-intro {color:var(--archive-secondary);margin:0 0 28px;font-size:15px;line-height:1.7;}
.tr-status {display:flex;align-items:center;gap:8px;color:var(--archive-secondary);font-size:13px;margin:12px 0;}
.tr-dot {width:7px;height:7px;border-radius:50%;background:var(--evidence-green);flex-shrink:0;}
.tr-dot.offline {background:var(--absence-amber);}
.tr-section-label {color:var(--archive-muted);font-size:12px;letter-spacing:.08em;margin:20px 0 8px;}
.tr-summary {display:flex;gap:32px;border-top:1px solid var(--archive-rule);border-bottom:1px solid var(--archive-rule);padding:20px 0;margin-bottom:24px;}
.tr-summary strong {font-size:24px;font-weight:650;font-variant-numeric:tabular-nums;}
.tr-summary span {display:block;font-size:13px;color:var(--archive-secondary);margin-top:4px;}
.tr-empty {padding:40px 24px;text-align:center;color:var(--archive-secondary);}
.tr-empty h3 {margin:16px 0 8px;font-size:20px!important;}
.tr-empty p {font-size:14px;color:var(--archive-secondary);max-width:420px;margin:auto;}
.tr-empty-symbol {display:inline-grid;place-items:center;width:64px;height:64px;background:var(--evidence-wash);border-radius:18px;color:var(--evidence-green);}
.tr-file {display:flex;align-items:center;gap:12px;min-width:0;padding:4px 0;}
.tr-file-type {flex-shrink:0;border:1px solid var(--archive-rule);background:var(--paper-inset);color:var(--evidence-green);border-radius:8px;padding:12px 8px;font-size:11px;font-weight:700;}
.tr-file-name {font-size:15px;font-weight:600;overflow-wrap:anywhere;}
.tr-file-meta {color:var(--archive-muted);font-size:12px;margin-top:4px;}
.tr-answer-label {color:var(--evidence-green);font-size:13px;font-weight:600;margin-bottom:12px;}
.tr-answer-label.refused {color:var(--absence-amber);}
.tr-source-number {display:inline-block;padding:3px 8px;background:var(--evidence-wash);color:var(--evidence-green);border-radius:5px;font-size:12px;font-weight:650;margin-bottom:8px;}
.tr-source-name {color:var(--archive-ink);font-weight:600;font-size:14px;overflow-wrap:anywhere;margin-bottom:4px;}
.tr-source-meta {color:var(--archive-muted);font-size:12px;margin-bottom:10px;}
.tr-question {font-size:18px;font-weight:600;line-height:1.7;overflow-wrap:anywhere;margin:4px 0 16px;}
.tr-footer {margin-top:32px;padding-top:16px;border-top:1px solid var(--archive-rule);color:var(--archive-muted);font-size:12px;}
@media(max-width:900px) {
 [data-testid="stMainBlockContainer"] {padding:4rem 1rem 3rem;}
 [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
 [data-testid="stHorizontalBlock"]>[data-testid="stColumn"] {min-width:100%!important;}
 .tr-heading {font-size:28px;} .tr-summary {gap:24px;}
}
@media(prefers-reduced-motion:reduce) {button {transition:none!important;}button:active {transform:none;}}
</style>
"""

ARCHIVE_MARK = '<svg width="21" height="21" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 3h11l3 3v15H6V3Z" stroke="white" stroke-width="1.5"/><path d="M9 9h8M9 13h8M9 17h5M3 7v14" stroke="white" stroke-width="1.5"/></svg>'
EMPTY_MARK = '<svg width="32" height="32" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 5h7l2 3h7v12H4V5Z" stroke="currentColor" stroke-width="1.4"/><path d="M8 12h8M8 16h5" stroke="currentColor" stroke-width="1.4"/></svg>'
