# ============================================================
# GERAÇÃO DE PDF DE EXEMPLO PARA TESTAR PyPDFLoader
# ============================================================

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

arquivo_pdf = "dados/manual_compras.pdf"

c = canvas.Canvas(arquivo_pdf, pagesize=A4)

c.drawString(80, 800, "Manual de Compras Corporativas")
c.drawString(80, 760, "Toda compra corporativa deve ser aprovada previamente.")
c.drawString(80, 730, "Compras sem aprovação podem ser recusadas pela área financeira.")
c.drawString(80, 700, "O processo deve conter justificativa, fornecedor e comprovante.")

c.showPage()

c.drawString(80, 800, "Regras para Fornecedores")
c.drawString(80, 760, "Fornecedores devem estar cadastrados antes da contratação.")
c.drawString(80, 730, "Pagamentos dependem de nota fiscal válida.")

c.save()

print(f"PDF criado com sucesso em: {arquivo_pdf}")