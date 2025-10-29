# Generative Adversarial Networks (GANs) con Fashion-MNIST 🧠👗

Este proyecto implementa **redes generativas adversarias (GANs)** desde cero utilizando **PyTorch**, para generar imágenes de ropa basadas en el dataset **Fashion-MNIST**.

Incluye dos variantes:
1. Un **GAN simple** con perceptrones multicapa (MLP).
2. Un **GAN convolucional (DCGAN)** que produce imágenes más realistas.

---

## 🚀 Características

- Descarga automática del dataset **FashionMNIST**.
- Implementación personalizada del `Dataset` y `DataLoader`.
- Entrenamiento tanto de un **generador** como de un **discriminador**.
- Visualización del proceso de entrenamiento y de las imágenes generadas.
- Gráficas de pérdida (`g_loss`, `d_loss`) para analizar la convergencia del modelo.

---

## 🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

```bash
pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
