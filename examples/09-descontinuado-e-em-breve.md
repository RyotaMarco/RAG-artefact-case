# 09. descontinuado-e-em-breve

**Scenario:** Unavailable items: discontinued with stock (Ibanez Bass 2X) and coming soon (Kala).

**What to check:** Neither is sold despite positive stock; the agent offers available alternatives.

*Recorded 2026-11-02T19:53-04:00 with `claude-haiku-5-5` in one new conversation.*

## Turn 1

**Customer:**

> Quero comprar o Ibanez Bass 2X. Vocês têm?

**Agent:**

Infelizmente o **Ibanez Bass 2X Electric Bass** não está disponível para compra no momento. Ele está descontinuado, então não consigo vender essa unidade, mesmo com estoque registrado.

Se você quer um baixo elétrico, estes estão ativos e em estoque:

- **Music Man Bass 1X Electric Bass**: R$ 8.732,00
- **Yamaha Bass 3X Electric Bass**: R$ 11.445,00
- **Music Man Bass 4X Electric Bass**: R$ 7.157,00

Quer que eu veja as especificações de algum deles ou calcule o valor no PIX?

**Tools called:**

- `query_data({"sql": "SELECT p.product_id, p.name, p.price_cents, p.stock_quantity, p.status, p.specs FROM products p WHERE p.search_name LIKE '%ibanezbass2x%' OR p.search_name LIKE '%ibanez%bass%2x%'"})`
- `query_data({"sql": "SELECT p.product_id, p.name, p.price_cents, p.stock_quantity FROM products p JOIN categories c ON c.category_id = p.category_id WHERE c.name = 'Baixos' AND p.status = 'active' AND p.stock_quantity > 0"})`

**Sources cited:**

- desafio_tecnico_ai_eng - products.csv
- desafio_tecnico_ai_eng - categories.csv

## Turn 2

**Customer:**

> E o Kala KA-ABP-CTG Barítono Cedar?

**Agent:**

O **Kala KA-ABP-CTG Barítono Cedar** ainda não está disponível para compra. Ele está marcado como **em breve**, então não consigo vender essa unidade, mesmo com 2 em estoque registradas.

Se você quer um ukulele de cedro com a mesma pegada natural, o mais próximo que temos ativo é o:

- **Kala KA-SRT-CTG Tenor Elétrico Cedar**: R$ 1.599,00, com 4 unidades em estoque. Tem tampo de cedro sólido, fundo e laterais de acácia, braço de mogno, cordas de nylon, escala de 432 mm e captação elétrica.

O barítono tem escala de 502 mm e fundo de pau-ferro, então o som e o tamanho são diferentes. Quer que eu calcule o valor no PIX do tenor elétrico ou veja outra opção?

**Tools called:**

- `query_data({"sql": "SELECT product_id, name, price_cents, status, stock_quantity, specs FROM products WHERE search_name LIKE '%kala%' AND search_name LIKE '%ctg%'"})`

**Sources cited:**

- desafio_tecnico_ai_eng - products.csv
