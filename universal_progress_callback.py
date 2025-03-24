import time
import threading
import logging
from typing import Dict, Any, Optional, Callable, List
from enum import Enum

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ProgressState(str, Enum):
    """Estados posibles del progreso RAG"""
    INITIALIZING = "initializing"
    QUERY_EXPANSION = "query_expansion"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    RESPONSE_GENERATION = "response_generation"
    CITATION_VALIDATION = "citation_validation"
    CITATION_EXTRACTION = "citation_extraction"
    FINAL_VALIDATION = "final_validation"
    COMPLETE = "complete"
    ERROR = "error"

# Mensajes descriptivos para cada estado
PROGRESS_MESSAGES = {
    ProgressState.INITIALIZING: "Inicializando el procesamiento de la consulta",
    ProgressState.QUERY_EXPANSION: "Expandiendo la consulta para mejorar los resultados",
    ProgressState.DOCUMENT_RETRIEVAL: "Recuperando documentos relevantes de la base de conocimiento",
    ProgressState.RESPONSE_GENERATION: "Generando una respuesta basada en los documentos recuperados",
    ProgressState.CITATION_VALIDATION: "Validando citas y referencias en la respuesta",
    ProgressState.CITATION_EXTRACTION: "Extrayendo información de citas para presentación",
    ProgressState.FINAL_VALIDATION: "Realizando validación final de la respuesta",
    ProgressState.COMPLETE: "¡Respuesta lista!",
    ProgressState.ERROR: "Error durante el procesamiento"
}

class UniversalProgressCallback:
    """
    Implementación universal de callback de progreso que puede usarse con 
    diferentes interfaces (CLI, GUI, API, etc.)
    """
    
    def __init__(self):
        self.current_state: ProgressState = ProgressState.INITIALIZING
        self.states_completed: List[ProgressState] = []
        self.details: Dict[str, Any] = {}
        self.start_time = time.time()
        self.on_progress_change: Optional[Callable[[ProgressState, Dict], None]] = None
        self.logger = logging.getLogger("RAGProgress")
        self.error: Optional[str] = None
        self._lock = threading.Lock()  # Para acceso thread-safe
    
    def register_callback(self, callback_fn: Callable[[ProgressState, Dict], None]) -> None:
        """
        Registra una función de callback que será llamada cuando cambie el progreso.
        
        Args:
            callback_fn: Función que acepta el estado actual y detalles adicionales
        """
        self.on_progress_change = callback_fn
        
    def update_progress(self, state_name: str, details: Optional[Dict] = None) -> None:
        """
        Actualiza el estado del progreso y notifica a los observadores.
        
        Args:
            state_name: Nombre del nuevo estado
            details: Detalles adicionales opcionales del estado
        """
        with self._lock:
            # Convertir el string a enum si es necesario
            if isinstance(state_name, str):
                try:
                    state = ProgressState(state_name)
                except ValueError:
                    self.logger.warning(f"Estado desconocido: {state_name}, usando 'initializing'")
                    state = ProgressState.INITIALIZING
            else:
                state = state_name
                
            # Actualizar estado actual
            self.current_state = state
            
            # Añadir a la lista de estados completados
            if state not in self.states_completed:
                self.states_completed.append(state)
            
            # Actualizar detalles
            if details:
                self.details.update(details)
            
            # Calcular tiempos y progreso
            elapsed_time = time.time() - self.start_time
            self.details["elapsed_time"] = elapsed_time
            
            # Calcular progreso estimado (0-100%)
            all_states = list(ProgressState)
            if state == ProgressState.ERROR:
                progress_percent = 100
            else:
                try:
                    current_index = all_states.index(state)
                    progress_percent = min(100, (current_index + 1) * 100 / len(all_states))
                except ValueError:
                    progress_percent = 0
                    
            self.details["progress_percent"] = progress_percent
            
            # Loggear el progreso
            self.logger.info(f"Progreso: {state.value} - {PROGRESS_MESSAGES.get(state)}")
            
            # Notificar al callback si existe
            if self.on_progress_change:
                try:
                    self.on_progress_change(state, self.details)
                except Exception as e:
                    self.logger.error(f"Error en callback de progreso: {str(e)}")
    
    def set_error(self, error_message: str) -> None:
        """
        Marca el proceso con un error.
        
        Args:
            error_message: Mensaje de error
        """
        with self._lock:
            self.error = error_message
            self.details["error"] = error_message
            self.update_progress(ProgressState.ERROR, {"error_message": error_message})
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen del progreso actual.
        
        Returns:
            Dict con un resumen del progreso
        """
        with self._lock:
            return {
                "current_state": self.current_state.value,
                "current_message": PROGRESS_MESSAGES.get(self.current_state),
                "progress_percent": self.details.get("progress_percent", 0),
                "elapsed_time": self.details.get("elapsed_time", 0),
                "states_completed": [s.value for s in self.states_completed],
                "error": self.error,
                "details": self.details
            }
    
    def reset(self) -> None:
        """Reinicia el progreso"""
        with self._lock:
            self.current_state = ProgressState.INITIALIZING
            self.states_completed = []
            self.details = {}
            self.start_time = time.time()
            self.error = None
            if self.on_progress_change:
                self.on_progress_change(self.current_state, self.details)


# Ejemplos de uso y adaptadores para diferentes interfaces

class CLIProgressAdapter:
    """Adaptador para mostrar el progreso en línea de comandos"""
    
    def __init__(self, callback: UniversalProgressCallback):
        self.callback = callback
        self.callback.register_callback(self.update_cli)
        self.spinner_chars = "|/-\\"
        self.spinner_idx = 0
    
    def update_cli(self, state: ProgressState, details: Dict) -> None:
        """Actualiza la visualización en CLI"""
        spinner = self.spinner_chars[self.spinner_idx % len(self.spinner_chars)]
        self.spinner_idx += 1
        
        elapsed = details.get("elapsed_time", 0)
        percent = details.get("progress_percent", 0)
        
        # Crear una barra de progreso sencilla
        progress_bar = self._create_progress_bar(percent)
        
        print(f"\r{spinner} {progress_bar} {percent:.1f}% - {PROGRESS_MESSAGES.get(state)} ({elapsed:.1f}s)", end="")
        
        if state == ProgressState.COMPLETE:
            print("\nProceso completado!")
        elif state == ProgressState.ERROR:
            print(f"\nError: {details.get('error_message', 'Error desconocido')}")
    
    def _create_progress_bar(self, percent: float, width: int = 20) -> str:
        """Crea una barra de progreso ASCII"""
        filled_width = int(width * percent / 100)
        return f"[{'#' * filled_width}{' ' * (width - filled_width)}]"


class APIProgressAdapter:
    """Adaptador para usar el progreso en un API"""
    
    def __init__(self, callback: UniversalProgressCallback):
        self.callback = callback
        # No necesitamos registrar un callback aquí, 
        # ya que el API simplemente consultará get_progress_summary()
    
    def get_progress_json(self) -> Dict:
        """Obtiene el progreso en formato JSON para un API"""
        return self.callback.get_progress_summary()


class SimpleLoggerAdapter:
    """Adaptador para logs simples"""
    
    def __init__(self, callback: UniversalProgressCallback, log_level: int = logging.INFO):
        self.callback = callback
        self.logger = logging.getLogger("RAGProgressLogger")
        self.logger.setLevel(log_level)
        self.callback.register_callback(self.log_progress)
    
    def log_progress(self, state: ProgressState, details: Dict) -> None:
        """Registra el progreso en los logs"""
        percent = details.get("progress_percent", 0)
        elapsed = details.get("elapsed_time", 0)
        self.logger.info(f"RAG Progress: {percent:.1f}% - {state.value} - {PROGRESS_MESSAGES.get(state)} ({elapsed:.1f}s)")
        
        if state == ProgressState.ERROR and "error_message" in details:
            self.logger.error(f"RAG Error: {details['error_message']}")


# Ejemplo de uso básico
if __name__ == "__main__":
    # Crear el callback
    progress = UniversalProgressCallback()
    
    # Crear adaptador para CLI
    cli_adapter = CLIProgressAdapter(progress)
    
    # Simular un proceso RAG
    def simulate_rag():
        states = [
            ProgressState.INITIALIZING,
            ProgressState.QUERY_EXPANSION,
            ProgressState.DOCUMENT_RETRIEVAL, 
            ProgressState.RESPONSE_GENERATION,
            ProgressState.CITATION_VALIDATION,
            ProgressState.CITATION_EXTRACTION,
            ProgressState.FINAL_VALIDATION,
            ProgressState.COMPLETE
        ]
        
        for state in states:
            time.sleep(1)  # Simular trabajo
            progress.update_progress(state)
    
    # Ejecutar la simulación
    print("Simulando proceso RAG...")
    simulate_rag()