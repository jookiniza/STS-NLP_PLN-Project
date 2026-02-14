# STS-NLP_PLN-Project
# Semantic Textual Similarity (STS) with Transformers

## Descripción del Proyecto
Este proyecto aborda el desafío de la Similitud Textual Semántica (STS), consistente en cuantificar el grado de equivalencia de significado entre pares de oraciones. Desarrollado como parte del Grado en Inteligencia Artificial en la Universidad del País Vasco, el estudio analiza la evolución desde modelos vectoriales clásicos hasta el estado del arte basado en Transformers.

## Metodología y Arquitectura
La investigación se estructura en tres niveles de complejidad creciente:

* **Z1 - Línea Base (Vectorial Clásica):** Implementación de un modelo Bag-of-Words con N-gramas y medición mediante Similitud Coseno. Esto sirve para establecer la brecha semántica respecto a los enfoques neuronales.
* **Z2 - Ajuste Fino Supervisado:** Entrenamiento de modelos BERT y RoBERTa utilizando una arquitectura **Cross-Encoder**. 
  * A diferencia de los Bi-Encoders, el Cross-Encoder permite que la atención cruce información entre ambas frases en todas las capas profundas.
  * La función de pérdida minimizada fue el Error Cuadrático Medio (MSE):
  $$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}(y_{i}-\hat{y}_{i})^{2}$$
  * Se aplicaron técnicas de estabilización como el optimizador AdamW, acumulación de gradientes y un Scheduler con Warmup para evitar el olvido catastrófico.
* **Z3 - Multilingüismo Zero-Shot:** Uso de XLM-RoBERTa para investigar la transferencia universal. 
  * El modelo fue entrenado exclusivamente en español y evaluado en inglés, ruso y chino.

## Resultados Clave

| Modelo | Correlación Pearson (r) |
| :--- | :--- |
| N-gramas + Coseno (Baseline) | 0.5705 |
| SBERT (Pre-entrenado) | 0.8274 |
| BERT-base (Fine-tuned) | 0.8465 |
| **RoBERTa-base (Fine-tuned)** | **0.8810** |

* **Superioridad de RoBERTa:** El modelo RoBERTa superó a BERT, demostrando que la eliminación de la tarea de predicción de la siguiente oración (NSP) y el enmascaramiento dinámico (Masking Dinámico) mejoran la generalización.
* **Transferencia Semántica Real:** Se logró transferir el conocimiento desde el español hacia el chino sin datos paralelos, alcanzando un rendimiento cercano a 0.79 en correlación. 
* **Alineación Isomórfica:** Este éxito en un idioma con logogramas demuestra que el modelo aprende a alinear espacios vectoriales abstractos, superando la simple ilusión de coincidencias de sub-palabras (BPE).

## Stack Tecnológico
* **Lenguaje:** Python
* **Librerías:** Hugging Face `transformers`
* **Hardware:** GPU NVIDIA T4
