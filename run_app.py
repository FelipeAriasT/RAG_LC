import os
import sys
import streamlit as st




# Agregar directorio raíz al path para poder importar utilidades
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Importar la aplicación Streamlit
from app.app import main

if __name__ == "__main__":
    # Ejecutar la aplicación
    main()