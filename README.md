# Modelamiento de un Perceptrón mediante Agentes en MESA - Johan Steven Galeano Gonzalez

## Introducción

El perceptrón es uno de los modelos más simples de redes neuronales artificiales. Su objetivo es aprender una frontera de decisión que separe correctamente dos clases de puntos en un plano bidimensional.  
En este proyecto se implementó un perceptrón usando el paradigma de agentes dentro del entorno **MESA** (Python), permitiendo observar de forma visual e interactiva el proceso de aprendizaje.

El propósito principal fue comprender cómo el perceptrón ajusta sus pesos durante el entrenamiento, y cómo los parámetros influyen en la velocidad y precisión del aprendizaje.

---
### Requisitos para ejecutar el programa

Para correr la simulación correctamente se necesita tener instalado:

- **Python 3.10 o superior**
- **Mesa 1.2.1 o superior**
- Un entorno virtual 
- Un navegador web 
- Editor, utilice visual studio code
## Funcionamiento del Perceptrón

<img width="652" height="389" alt="image" src="https://github.com/user-attachments/assets/86c2e19c-cd2b-4779-8139-435d64fc3929" />


---

## Implementación en MESA

La implementación se desarrolló en Python con el framework **MESA**, empleando un diseño basado en agentes:

- **Agente `DataPoint`:** representa un punto en el plano con coordenadas (x, y) y una clase asociada (+1 o -1).  
- **Modelo `PerceptronModel`:** se encarga de generar los puntos, calcular las predicciones, ajustar los pesos y medir la precisión.  
- **Interfaz visual:** permite manipular los parámetros del modelo mediante controles tipo *slider*:
  - Tasa de aprendizaje (η)
  - Número de iteraciones (épocas)
  - Tamaño del conjunto de datos (N)

Durante la simulación:
- Los puntos correctamente clasificados se muestran en color **verde**.  
- Los puntos mal clasificados se muestran en color **rojo**.  
- La **línea gris** representa la frontera de decisión que el modelo ajusta a medida que aprende.

---

## Resultados experimentales

Se realizaron varios experimentos modificando los parámetros del modelo para observar su comportamiento.

| Tasa de aprendizaje (η) | Épocas | Dataset | Exactitud entrenamiento | Exactitud prueba | Frontera de decisión |
|--------------------------|---------|----------|-------------------------|------------------|----------------------|
| 0.01 | 50 | 200 | 98.5% | 95.5% | y = -0.19x + 0.13 |
| 0.10 | 50 | 200 | 100% | 100% | y = 1.28x + 0.95 |
| 1.00 | 60 | 400 | 100% | 100% | y = 0.53x + 1.18 |


Los resultados se almacenaron automáticamente en el archivo `metrics.csv`, que contiene las métricas por época: precisión de entrenamiento y valores de los pesos (w₀, w₁, w₂).

Tasa de aprendizaje 0.01 50 epocas 200 datasets 
<img width="921" height="516" alt="image" src="https://github.com/user-attachments/assets/38f69ec2-3173-4798-b8a2-2669e37bae11" />


Tasa de aprendizaje 0.1 50 epocas 200 datasets 
<img width="921" height="522" alt="image" src="https://github.com/user-attachments/assets/cb02f679-9802-446c-aa2e-1c10f4593e87" />

Tasa de aprendizaje 1, 60 epocas 400 datasets 
<img width="921" height="595" alt="image" src="https://github.com/user-attachments/assets/91a7fbdb-44eb-40b9-a0e4-013a039aa8fe" />


---

## Análisis de los resultados

- Con **η = 0.01**, el aprendizaje fue lento y progresivo, mostrando un aumento gradual en la precisión.  
- Con **η = 0.10**, el modelo alcanzó rápidamente el 100% de exactitud sin inestabilidad.  
- Con **η = 1.00**, la convergencia también fue completa debido a que los datos son linealmente separables, aunque en datos reales una tasa tan alta podría generar oscilaciones.

Cuando el perceptrón logra clasificar todos los puntos correctamente, deja de modificar los pesos, cumpliendo el **Teorema de Convergencia del Perceptrón**, que garantiza la existencia de una solución perfecta si los datos son separables por una línea recta.

---

## Conclusiones

- El modelo de perceptrón implementado clasifica correctamente los datos generados, alcanzando un 100% de precisión en conjuntos linealmente separables.  
- La tasa de aprendizaje afecta directamente la velocidad de convergencia y la estabilidad del entrenamiento.  
- A través de la simulación en MESA, es posible visualizar el proceso de ajuste de pesos y entender de forma intuitiva el funcionamiento interno del algoritmo.  
- Este proyecto demuestra la utilidad del enfoque basado en agentes para enseñar conceptos fundamentales de aprendizaje automático.

---

