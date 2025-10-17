from __future__ import annotations
# Importaciones de Mesa y librerias necesarias
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.modules import TextElement
from mesa.visualization.UserParam import UserSettableParameter
from model import PerceptronModel


# Muestra texto con las metricas del modelo (epoca, accuracy y linea)
class AccuracyText(TextElement):
    def render(self, model: PerceptronModel):
        acc = model.train_accuracy * 100
        line = model.decision_line()
        test = model.test_accuracy
        msg = f"Epoca: {model.current_epoch}/{model.iterations} | Acc. entrenamiento: {acc:.1f}%"
        if model.training_done and test is not None:
            msg += f" | Acc. prueba: {test*100:.1f}%"
        if isinstance(line, dict) and "a" in line:
            msg += f" | Frontera: y = {line['a']:.2f}x + {line['b']:.2f}"
        elif isinstance(line, dict) and line.get("vertical") is not None:
            msg += f" | Frontera: x = {line['vertical']:.2f}"
        return msg


# Canvas para dibujar los puntos y la linea de decision
class ScatterCanvas(VisualizationElement):
    package_includes = []
    local_includes = []

    def __init__(self, width=700, height=520):
        super().__init__()
        self.width = width
        self.height = height

        # Codigo JavaScript que dibuja el grafico en el navegador
        self.js_code = f"""
            (function() {{
                class ScatterModule {{
                    constructor(width, height) {{
                        this.width = width;
                        this.height = height;

                        // Contenedor visual
                        const wrapper = document.createElement('div');
                        wrapper.style.display = 'block';
                        wrapper.style.margin = '12px auto';
                        wrapper.style.width = (width + 2) + 'px';

                        // Titulo
                        const title = document.createElement('div');
                        title.textContent = 'Visualizacion (puntos y frontera)';
                        title.style.font = '600 14px system-ui, -apple-system, Segoe UI, Roboto';
                        title.style.margin = '6px 0';
                        wrapper.appendChild(title);

                        // Lienzo
                        this.canvas = document.createElement('canvas');
                        this.canvas.width = width;
                        this.canvas.height = height;
                        this.canvas.style.border = '1px solid #888';
                        this.canvas.style.background = '#f3f3f3';
                        wrapper.appendChild(this.canvas);

                        this.ctx = this.canvas.getContext('2d');
                        this.el = wrapper;
                    }}

                    reset() {{
                        if (this.ctx) this.ctx.clearRect(0, 0, this.width, this.height);
                    }}

                    render(data) {{
                        const ctx = this.ctx;
                        ctx.clearRect(0, 0, this.width, this.height);

                        // Ejes de referencia
                        ctx.strokeStyle = '#e0e0e0';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(0, this.height/2); ctx.lineTo(this.width, this.height/2);
                        ctx.moveTo(this.width/2, 0);  ctx.lineTo(this.width/2, this.height);
                        ctx.stroke();

                        // Dibuja puntos (verde = correcto, rojo = incorrecto)
                        for (const p of data.points) {{
                            const x = p.x * this.width;
                            const y = (1 - p.y) * this.height;
                            ctx.beginPath();
                            ctx.arc(x, y, 4, 0, 2*Math.PI);
                            ctx.fillStyle = p.correct ? '#2ecc71' : '#e74c3c';
                            ctx.fill();
                            ctx.lineWidth = 1.5;
                            ctx.strokeStyle = (p.label === 1) ? '#2980b9' : '#8e44ad';
                            ctx.stroke();
                        }}

                        // Dibuja la linea de decision
                        if (data.line) {{
                            ctx.lineWidth = 2;
                            ctx.strokeStyle = '#34495e';
                            ctx.beginPath();
                            if (data.line.vertical !== undefined && data.line.vertical !== null) {{
                                const x = data.line.vertical * this.width;
                                ctx.moveTo(x, 0);
                                ctx.lineTo(x, this.height);
                            }} else if (data.line.a !== undefined && data.line.b !== undefined) {{
                                const cand = [
                                    [0, data.line.b],
                                    [1, data.line.a + data.line.b],
                                    [-(data.line.b)/(data.line.a || 1e-9), 0],
                                    [-(data.line.b-1)/(data.line.a || 1e-9), 1]
                                ];
                                const pts = [];
                                for (const [xx, yy] of cand) {{
                                    if (xx >= 0 && xx <= 1 && yy >= 0 && yy <= 1) pts.push([xx, yy]);
                                }}
                                if (pts.length >= 2) {{
                                    const [xa, ya] = pts[0];
                                    const [xb, yb] = pts[1];
                                    ctx.moveTo(xa * this.width, (1 - ya) * this.height);
                                    ctx.lineTo(xb * this.width, (1 - yb) * this.height);
                                }}
                            }}
                            ctx.stroke();
                        }}
                    }}

                    getElement() {{ return this.el; }}
                }}

                // Agrega el canvas al sistema de Mesa
                function mountWhenReady() {{
                    if (typeof elements !== 'undefined' && Array.isArray(elements)) {{
                        const mod = new ScatterModule({self.width}, {self.height});
                        elements.push(mod);

                        const tryMount = () => {{
                            const host = document.getElementById('elements')
                                     || document.querySelector('#elements')
                                     || document.body;
                            if (host && typeof mod.getElement === 'function') {{
                                if (!mod.el.parentNode) host.appendChild(mod.getElement());
                            }} else {{
                                setTimeout(tryMount, 50);
                            }}
                        }};
                        tryMount();
                    }} else {{
                        setTimeout(mountWhenReady, 50);
                    }}
                }}
                mountWhenReady();
            }})();
        """

    # Devuelve los datos al canvas (puntos y linea)
    def render(self, model: PerceptronModel):
        return {
            "points": [{"x": a.x, "y": a.y, "label": a.label, "correct": a.correct}
                       for a in model.schedule.agents],
            "line": model.decision_line()
        }


# Crea el servidor y los sliders de configuracion
def make_server():
    canvas = ScatterCanvas(700, 520)
    info = AccuracyText()
    lr = UserSettableParameter("slider", "Tasa de aprendizaje (η)", 0.1, 0.01, 1.0, 0.01)
    it = UserSettableParameter("slider", "Iteraciones (epocas)", 50, 1, 300, 1)
    n  = UserSettableParameter("slider", "Tamaño del dataset (N)", 200, 20, 1000, 10)
    params = {"learning_rate": lr, "iterations": it, "N": n}

    server = ModularServer(
        PerceptronModel,
        [canvas, info],  # elementos visuales
        "Perceptron con MESA — Clasificacion 2D",
        params
    )
    server.port = 8521
    return server


# Ejecuta la aplicacion
if __name__ == "__main__":
    make_server().launch()
