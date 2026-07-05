# GridGuard España — Guion del vídeo de entrega (≤5 min, YouTube público)

Documento de trabajo interno para grabar el vídeo del capstone. No se sube al repo como
entregable en sí — es tu apoyo para grabar. Cronometrado a **4:45** para dejar margen de 15s.

**Regla de oro para grabar:** no intentes correr `run_scenario()` completo EN DIRECTO durante
la grabación — es reproduciblemente inestable ahora mismo (ver `eval/README.md`). Todo lo que
se muestra en pantalla en este guion o (a) es determinista y siempre funciona, o (b) es una
captura/output ya verificado de una corrida anterior.

---

## 0:00–0:45 — Problema (45s)

**Qué decir (guion, adáptalo a tu voz):**

> "El 28 de abril de 2025, España sufrió un apagón real de 6 horas y media que afectó a 15.000
> megavatios. El informe oficial de REE muestra que las señales de aviso estaban ahí, horas
> antes: penetración renovable extrema, inercia síncrona cayendo, balance oferta-demanda
> empeorando progresivamente. Pero hay un problema: **la frecuencia de la red — la señal más
> directa — no se publica en ninguna API pública.** REE no la expone por seguridad operacional.
> Este proyecto pregunta: ¿puede un sistema multi-agente, trabajando solo con lo que sí está
> disponible públicamente, detectar esa situación a tiempo?"

**Qué mostrar en pantalla:** el título del proyecto + una imagen/mapa de España (o la portada
que generes), sin necesidad de terminal todavía.

---

## 0:45–1:30 — Por qué agentes (45s)

**Qué decir:**

> "GridGuard combina tres tipos de inteligencia, cada una solo donde de verdad hace falta:
> Python puro para lo que es determinista, Machine Learning para detectar patrones
> estadísticos — un ensemble de Isolation Forest, un autoencoder LSTM y un clasificador
> calibrado — y una cadena de Markov que proyecta hacia dónde va la red. El LLM solo entra
> cuando hace falta razonamiento real: sintetizar señales, consultar normativa, decidir cómo
> escalar. Este principio se llama shift-left, y lo aplicamos incluso retroactivamente: cuando
> descubrimos que el LLM no seguía fielmente la política de escalación, la solución no fue
> 'mejorar el prompt' — fue quitarle esa decisión al LLM por completo y dársela a una función
> Python determinista con 17 tests."

**Qué mostrar en pantalla:** la tabla de "3 niveles de inteligencia" (puedes leerla de
`README.md` o `MEMORIA.md` §3, o mostrarla como slide).

---

## 1:30–2:45 — Arquitectura (75s)

**Qué decir:**

> "Seis capas. Los triggers arrancan el ciclo. La capa 1 son Function Tools puros — sin LLM.
> La capa 2 son 5 MCP servers custom: datos de red, meteorología, memoria persistente, modelos
> ML, y una base de conocimiento RAG con normativa REE y ENTSO-E. La capa 3 son Skills — agentes
> envueltos como herramientas síncronas, cada uno con un único resultado estructurado. La capa 4
> son sub-agentes con su propio bucle — uno redacta informes de emergencia, otro evalúa
> importación de energía transfronteriza. Y arriba de todo, el Orchestrator, el agente raíz que
> coordina todo esto y decide cómo escalar.
>
> Todo esto corre en local — no lo desplegamos en la nube por coste y por tiempo, dado el plazo
> del proyecto. En producción 24/7 cambiaría la infraestructura, no la lógica de agentes: SQLite
> pasaría a Cloud SQL, ChromaDB local a un vector store gestionado, y los MCP servers de
> subprocesos a servicios persistentes en Cloud Run."

**Qué mostrar en pantalla:** el diagrama de arquitectura (`architecture_diagram.html`,
actualizado) o el diagrama ASCII de capas de `README.md`/`ARCHITECTURE.md`.

**Nota de tiempo:** esta coda de deployability añade ~10s a los 75s de esta sección. Si al
grabar te queda apretado, es la frase más fácil de recortar (o acortarla a "en producción
cambiaría la infraestructura, no la lógica de agentes — el README tiene el detalle").

---

## 2:45–4:00 — Demo (75s)

**Opción A (recomendada, sin riesgo — grábala primero, es la garantía del vídeo):**

Muestra la suite de tests offline corriendo en vivo, en tu terminal:

```bash
cd "ruta al proyecto"
python -m pytest -q
```

**Qué decir mientras corre (tarda ~15-20s, tiempo real para narrar):**

> "300 tests offline, deterministas, sin llamar a ningún API — corren en segundos y validan
> cada capa: la fórmula de riesgo, la política de escalación, los guardrails de seguridad, la
> memoria de lecciones aprendidas... todo lo que no necesita un LLM está probado sin depender
> de uno."

**Opción B (opcional, solo si quieres arriesgarte a algo en vivo — NO como única prueba):**

Correr un test live de UN agente aislado (no el orchestrator completo), por ejemplo:

```bash
python -m pytest eval/agents/regulatory_compliance -m integration -v
```

Esto consume cuota Gemini real y tarda ~30-60s — pruébalo TÚ ANTES de grabar para confirmar que
pasa en ese momento (la flakiness de LLM es real, ver `eval/README.md`). Si falla al probarlo,
usa solo la Opción A y complementa hablando de los resultados ya documentados: "en nuestra
validación, los agentes individuales — cumplimiento normativo, previsión renovable — pasan de
forma consistente contra Gemini real, y seguimos endureciendo la fiabilidad de la cadena
completa del orquestador, con cada hallazgo documentado en el repo."

**Qué mostrar en pantalla:** terminal con el resultado (verde), y opcionalmente el archivo
`eval/README.md` abierto en la sección de hallazgos, para mostrar que la validación fue real
y rigurosa, no solo "tests que siempre pasan".

---

## 4:00–4:45 — Cómo se construyó (45s)

**Qué decir:**

> "Se construyó con Google ADK 2.3.0 y Gemini. El desarrollo se apoyó en otro asistente de
> código con IDE propio, ya contratado de antes — la cuota gratuita de la API de Gemini se
> había agotado, así que reservamos cada token de pago para lo que de verdad lo necesitaba:
> probar los agentes contra el modelo real, no planificar o escribir código. Cada feature
> siguió un proceso de diseño → especificación escrita → plan de implementación →
> ejecución con revisión independiente en cada paso — no solo 'código que funciona', sino
> decisiones documentadas y trazables. Cuando la validación en vivo reveló comportamiento
> inesperado del LLM, no lo ignoramos: usamos herramientas de diagnóstico como py-spy para
> encontrar causas raíz reales — un import bloqueante, una excepción de terceros sin capturar,
> un modelo que no obedecía su propia política — y corregimos cada una con evidencia, no con
> parches a ciegas. Lo que seguimos puliendo también queda documentado, con la misma
> honestidad."

**Qué mostrar en pantalla:** `docs/superpowers/specs/` y `docs/superpowers/plans/` (la carpeta,
para mostrar volumen — 8 specs, 13 planes), o `MEMORIA.md` abierto en la sección de hallazgos.

**Nota de tiempo:** con la frase añadida sobre la herramienta de desarrollo, este bloque ronda
ahora los 55-60s hablados, no 45s — sumado a los ~10s añadidos en Arquitectura, el guion se
pasaría de 5:00 si lo lees todo tal cual. Recorta al grabar: la frase más prescindible es
"usamos herramientas de diagnóstico como py-spy para encontrar causas raíz reales" (ya está en
el README/MEMORIA, no aporta nada nuevo en voz) — quitarla sola ya recupera ~8s.

---

## 4:45–5:00 — Cierre (15s)

**Qué decir:**

> "GridGuard España: seis capas de Python, ML y agentes Gemini vigilando la red eléctrica
> española en tiempo real, con red de seguridad propia para los momentos en que el LLM no se
> comporta. Código y documentación completos en el repositorio."

**Qué mostrar en pantalla:** el README.md del repo, o la pantalla de título de cierre.

---

## Checklist antes de grabar

- [x] Diagrama de arquitectura actualizado y verificado visualmente (2026-07-05)
- [ ] Repo de GitHub ya en público (si no, el link del vídeo/writeup no servirá)
- [ ] `python -m pytest -q` probado justo antes de grabar (confirma 300 passed)
- [ ] Si vas a usar la Opción B del demo, probarla ANTES de grabar y tener un plan B si falla
- [ ] Grabación de pantalla lista (Win+G, o tu grabador habitual) + audio de tu voz o TTS
- [ ] Vídeo final ≤ 5:00 exactos, subido a YouTube como **público** (no "no listado")
