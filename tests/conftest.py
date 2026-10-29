"""
Configuración de pytest para el proyecto.

Asegura que la raíz del proyecto esté en ``sys.path`` para que los tests
puedan importar el paquete ``functions`` y el módulo ``config`` sin
instalar el proyecto.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Raíz del proyecto (dos niveles arriba de tests/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
