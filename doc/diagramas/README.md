# Diagramas del proyecto

## Archivos

| Archivo | Tipo | Herramienta |
|---|---|---|
| `arquitectura_sistema.drawio` | Diagrama de componentes | PlantUML |
| `casos_de_uso.puml` | Diagrama de casos de uso | PlantUML |
| `secuencia_ciclo_dqn.puml` | Diagrama de secuencia | PlantUML |
| `modelo_entidad_relacion.drawio` | MER conceptual | PlantUML |
| `historias_usuario.md` | Historias de usuario | Markdown |

## Cómo renderizar los archivos .puml

### Opción 1 — VS Code (recomendada)
1. Instalar extensión **PlantUML** (jebbs.plantuml)
2. Abrir el `.puml` y pulsar `Alt+D` para previsualizar
3. Click derecho → "Export Current Diagram" → PNG o SVG

### Opción 2 — Online
Pegar el contenido en [plantuml.com/plantuml](https://www.plantuml.com/plantuml/uml/)

### Opción 3 — CLI
```bash
java -jar plantuml.jar arquitectura_sistema.puml
```
