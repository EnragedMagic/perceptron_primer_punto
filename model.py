from __future__ import annotations
from mesa import Model, Agent
from mesa.space import ContinuousSpace
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from pathlib import Path
import random

# Clase que representa cada punto de datos
class DataPoint(Agent):
    def __init__(self, unique_id, model, x, y, label):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.label = label      # clase (+1 o -1)
        self.correct = False    # indica si fue clasificado bien

    def step(self):
        pass  # no hace nada, el modelo entrena


# Modelo principal del perceptron
class PerceptronModel(Model):
    def __init__(self, N=200, learning_rate=0.1, iterations=50, seed=None):
        super().__init__(seed=seed)
        self.N = int(N)                     # cantidad de puntos
        self.learning_rate = float(learning_rate)  # tasa de aprendizaje
        self.iterations = int(iterations)   # epocas
        self.space = ContinuousSpace(1.0, 1.0, torus=False)  # plano de trabajo
        self.schedule = RandomActivation(self)  # actualizacion aleatoria
        self.current_epoch = 0
        self.training_done = False

        self._gen_true_boundary()  # crea la linea real de separacion

        # pesos iniciales aleatorios
        self.w0 = random.uniform(-1, 1)
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)

        self._generate_points()    # crea los puntos
        self._update_predictions() # calcula clasificaciones iniciales

        # variables para metricas
        self.train_accuracy = 0.0
        self.test_accuracy = None
        self._dumped = False

        # recolector de datos para graficas
        self.datacollector = DataCollector(
            model_reporters={
                "epoch": lambda m: m.current_epoch,
                "train_accuracy": lambda m: m.train_accuracy,
                "w0": lambda m: m.w0,
                "w1": lambda m: m.w1,
                "w2": lambda m: m.w2,
            }
        )

    # genera la linea real para separar clases
    def _gen_true_boundary(self):
        self.true_a = random.uniform(-2.5, 2.5)
        self.true_b = random.uniform(-0.6, 0.6)

    # asigna etiquetas segun la linea real
    def _label_by_true_line(self, x, y):
        return 1 if (y - (self.true_a * x + self.true_b)) >= 0 else -1

    # crea puntos con sus etiquetas
    def _generate_points(self):
        self.schedule = RandomActivation(self)
        self.space._index = {}
        self.unique_id_counter = 0
        for _ in range(self.N):
            x = random.uniform(0.02, 0.98)
            y = random.uniform(0.02, 0.98)
            label = self._label_by_true_line(x, y)
            a = DataPoint(self.unique_id_counter, self, x, y, label)
            self.unique_id_counter += 1
            self.space.place_agent(a, (x, y))
            self.schedule.add(a)

    # calcula salida sin activar
    def _predict_raw(self, x, y):
        return self.w0 + self.w1 * x + self.w2 * y

    # aplica funcion de activacion
    def _predict_label(self, x, y):
        return 1 if self._predict_raw(x, y) >= 0 else -1

    # actualiza clasificacion y precision
    def _update_predictions(self):
        correct = 0
        for agent in self.schedule.agents:
            yhat = self._predict_label(agent.x, agent.y)
            agent.correct = (yhat == agent.label)
            if agent.correct:
                correct += 1
        self.train_accuracy = correct / max(1, len(self.schedule.agents))

    # realiza una epoca de entrenamiento
    def _one_epoch(self):
        agents = self.schedule.agents[:]
        random.shuffle(agents)
        for agent in agents:
            x, y, t = agent.x, agent.y, agent.label
            yhat = 1 if self._predict_raw(x, y) >= 0 else -1
            if yhat != t:
                self.w0 += self.learning_rate * t * 1.0
                self.w1 += self.learning_rate * t * x
                self.w2 += self.learning_rate * t * y
        self._update_predictions()

    # controla las epocas y guarda resultados
    def step(self):
        if not self.training_done:
            if self.current_epoch < self.iterations:
                self._one_epoch()
                self.current_epoch += 1
                if self.current_epoch == self.iterations:
                    self.training_done = True
                    self._evaluate_test_set()
                    if not self._dumped:
                        self._dump_metrics_csv("metrics.csv")
                        self._dumped = True
            else:
                self.training_done = True
        self.datacollector.collect(self)

    # prueba con nuevos puntos
    def _evaluate_test_set(self, M: int = 200):
        correct = 0
        for _ in range(M):
            x = random.uniform(0.0, 1.0)
            y = random.uniform(0.0, 1.0)
            true_label = self._label_by_true_line(x, y)
            pred = self._predict_label(x, y)
            if pred == true_label:
                correct += 1
        self.test_accuracy = correct / M

    # exporta metricas a CSV
    def _dump_metrics_csv(self, path: str = "metrics.csv"):
        try:
            df = self.datacollector.get_model_vars_dataframe()
            df.to_csv(path, index=False)
            print(f"[INFO] Metricas exportadas a {Path(path).resolve()}")
        except Exception as e:
            print(f"[WARN] No se pudo exportar metricas: {e}")

    # calcula la linea de decision actual
    def decision_line(self):
        eps = 1e-9
        if abs(self.w2) < eps:
            if abs(self.w1) < eps:
                return {"vertical": None}
            return {"vertical": float(-self.w0 / self.w1)}
        a = - self.w1 / self.w2
        b = - self.w0 / self.w2
        return {"a": float(a), "b": float(b)}

