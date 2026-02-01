# src/config.py
import os

# Configurações do modelo
MODEL_CONFIG = {
    "path": "models/mistral-7b-instruct.gguf",
    "context_size": 2048,
    "threads": 8,  # Ajuste baseado no seu CPU
    "gpu_layers": 0,  # 0 = CPU, >0 = GPU
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 512
}

# Configurações do sistema
SYSTEM_CONFIG = {
    "name": "Mistral 7B Local",
    "version": "1.0.0",
    "description": "LLM local usando Mistral 7B Instruct"
}

def check_model():
    """Verifica se o modelo existe"""
    if not os.path.exists(MODEL_CONFIG["path"]):
        print(f"❌ Modelo não encontrado em: {MODEL_CONFIG['path']}")
        print("Por favor, coloque o modelo na pasta 'models/'")
        return False
    return True

def get_system_info():
    """Obtém informações do sistema"""
    import platform
    import multiprocessing
    
    return {
        "system": platform.system(),
        "processor": platform.processor(),
        "cores": multiprocessing.cpu_count(),
        "python_version": platform.python_version()
    }
