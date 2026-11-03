# 08. promocoes

**Scenario:** Promotions: inactive (Takamine GD20) vs active (Ohana CK-20, 20%) and PIX stacking.

**What to check:** Inactive promotion not offered; active price shown with original price and percentage; PIX 5% does not stack on a promotional price; a two-product order is quoted in one call, with shipping on the order total.

*Recorded 2026-11-02T19:53-04:00 with `claude-haiku-5-5` in one new conversation.*

## Turn 1

**Customer:**

> O Takamine GD20 está em promoção? E o Ohana CK-20?

**Agent:**

**Takamine GD20 Dreadnought Natural:** não está em promoção no momento. O preço é **R$ 2.199,00**, com 5 unidades em estoque.

**Ohana CK-20 Concert Natural:** está em promoção, com 20% de desconto.
- Preço original: **R$ 549,00**
- Preço promocional: **R$ 439,20**
- Estoque: 6 unidades

Essa promoção não se soma ao desconto do PIX. Se quiser, posso calcular o valor final de um pedido com os dois itens ou de cada um separadamente.

**Tools called:**

- `query_data({"sql": "SELECT p.product_id, p.name, p.price_cents, p.status, p.stock_quantity, pr.discount_percent, pr.is_active, pr.description FROM products p LEFT JOIN promotions pr ON pr.product_id = p.product_id WHERE p.search_name LIKE '%takamine%gd20%' OR p.search_name LIKE '%ohana%ck20%'"})`
- `calculate_quote({"items": [{"product_id": 95, "quantity": 1}, {"product_id": 121, "quantity": 1}]})`

**Sources cited:**

- desafio_tecnico_ai_eng - products.csv
- desafio_tecnico_ai_eng - promotions.csv
- políticas_da_loja.pdf · 3. Formas de Pagamento · pages (3,)
- políticas_da_loja.pdf · 3.1 Regras de Parcelamento · pages (3,)
- políticas_da_loja.pdf · 6.2 Regras de Promoções · pages (6,)
- políticas_da_loja.pdf · 5.1 Entregas na Região Metropolitana de Campo Grande · pages (4,)

## Turn 2

**Customer:**

> Se eu pagar o Ohana no PIX, ganho mais 5% de desconto?

**Agent:**

Não, o desconto de 5% do PIX não se aplica ao Ohana CK-20. Ele já está com a promoção de 20%, e a política da loja não permite somar os dois descontos.

No PIX, o valor fica assim:
- Preço original: **R$ 549,00**
- Preço com a promoção (PIX): **R$ 439,20**

Esse valor é só informativo, não é uma compra. Se você for da região metropolitana de Campo Grande, o frete é **R$ 35,00**, o que dá **R$ 474,20** no total. Para outras cidades, o frete depende do CEP e a equipe passa o valor pelo WhatsApp **(67) 3321-4500**.

**Tools called:**

- `query_data({"sql": "SELECT product_id, name, price_cents, stock_quantity, status FROM products WHERE search_name LIKE '%ohanack20%'"})`
- `calculate_quote({"items": [{"product_id": 121, "quantity": 1}], "payment_method": "pix"})`

**Sources cited:**

- desafio_tecnico_ai_eng - products.csv
- desafio_tecnico_ai_eng - promotions.csv
- políticas_da_loja.pdf · 3. Formas de Pagamento · pages (3,)
- políticas_da_loja.pdf · 3.1 Regras de Parcelamento · pages (3,)
- políticas_da_loja.pdf · 6.2 Regras de Promoções · pages (6,)
- políticas_da_loja.pdf · 5.1 Entregas na Região Metropolitana de Campo Grande · pages (4,)

## Turn 3

**Customer:**

> E se eu levar o Ohana CK-20 e o Shelby SN-100 juntos no PIX, entregando em Campo Grande, quanto fica com o frete?

**Agent:**

Para os dois instrumentos no PIX, entregando em Campo Grande, o **frete é R$ 0,00**. O total fica em **R$ 960,75**.

| Item | Preço original | Preço atual |
|---|---|---|
| Ohana CK-20 Concert Natural | R$ 549,00 | R$ 439,20 (promoção de 20%) |
| Shelby SN-100 Folk Aço Preto | R$ 549,00 | sem promoção ativa |

- **Total do pedido (sem frete):** R$ 988,20 com os preços atuais, e R$ 960,75 no PIX.
- O desconto do PIX não se soma à promoção do Ohana, então ele fica em R$ 439,20.
- O frete para a região metropolitana de Campo Grande é calculado sobre o valor final pago, por isso aparece como R$ 0,00.

Esse valor é só uma estimativa e não reserva os itens. Os dois têm estoque disponível agora.

**Tools called:**

- `query_data({"sql": "SELECT product_id, name, price_cents, stock_quantity, status FROM products WHERE search_name LIKE '%ohanack20%' OR search_name LIKE '%shelbysn100%'"})`
- `calculate_quote({"items": [{"product_id": 121, "quantity": 1}, {"product_id": 89, "quantity": 1}], "payment_method": "pix"})`

**Sources cited:**

- desafio_tecnico_ai_eng - products.csv
- desafio_tecnico_ai_eng - promotions.csv
- políticas_da_loja.pdf · 3. Formas de Pagamento · pages (3,)
- políticas_da_loja.pdf · 3.1 Regras de Parcelamento · pages (3,)
- políticas_da_loja.pdf · 6.2 Regras de Promoções · pages (6,)
- políticas_da_loja.pdf · 5.1 Entregas na Região Metropolitana de Campo Grande · pages (4,)
