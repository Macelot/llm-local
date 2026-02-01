---
noteId: "d618dfa0ff7b11f09255056be076a49d"
tags: []

---

# 1. Criar pasta do projeto
mkdir llm_local
cd llm_local

# 2. Criar ambiente virtual
python3 -m venv venv

# 3. Ativar ambiente (Linux/Mac)
source venv/bin/activate

# 4. Ativar ambiente (Windows)
venv\Scripts\activate

# 5. Criar arquivo requirements.txt
llama-cpp-python==0.2.45
numpy>=1.24.0
transformers>=4.35.0
sentence-transformers>=2.2.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0

# 6. Instalar dependências
pip install -r requirements.txt

# 7. Criar pasta para modelos
mkdir models

# 8. Baixar e Mover seu modelo (https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_0.gguf) 
# para a pasta models
mv /caminho/do/mistral-7b-instruct.gguf models/

# 9. Criar codigo Python para CHAT 
# llm_chat.py - 30 linhas
from llama_cpp import Llama

model = Llama(
    model_path="models/mistral-7b-instruct.gguf",
    n_ctx=2048,
    verbose=False
)

def chat():
    print("Chat com Mistral 7B (digite 'sair')")
    
    while True:
        # 6-9: Pegar input
        user_input = input("\nVocê: ").strip()
        if user_input.lower() == 'sair':
            break
        
        # 10-14: Gerar resposta
        response = model(
            f"[INST] {user_input} [/INST]",
            max_tokens=200,
            temperature=0.7
        )
        
        # 15: Mostrar resposta
        print(f"Mistral: {response['choices'][0]['text'].strip()}")

if __name__ == "__main__":
    chat()


# FEITO, basta rodar no terminal
python llm_chat.py    



# Se desejar aprimorar, podemos separar os códigos em 3 etapas: 10, 11 e 12
# 10. Criar codigo Python para juntar tudo 
#src/main.py
from llama_cpp import Llama
import time

class LocalLLM:
    def __init__(self, model_path: str):
        """
        Inicializa o modelo Mistral 7B
        """
        print(f"Carregando modelo: {model_path}")
        start_time = time.time()
        
        # Configurações do modelo
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,           # Contexto máximo (tokens)
            n_threads=8,          # Número de threads
            n_gpu_layers=0,       # 0 = CPU only, altere se tiver GPU
            verbose=True
        )
        
        load_time = time.time() - start_time
        print(f"Modelo carregado em {load_time:.2f} segundos")
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Gera texto baseado no prompt
        """
        try:
            # Configurar o prompt no formato Mistral
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            # Gerar resposta
            response = self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                echo=False,
                stop=["[INST]", "[/INST]", "\n\n"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            return f"Erro: {str(e)}"
    
    def chat(self, system_prompt: str = "Você é um assistente útil."):
        """
        Modo conversação interativa
        """
        print("\n" + "="*50)
        print("Chat com Mistral 7B Local")
        print("Digite 'sair' para encerrar")
        print("="*50 + "\n")
        
        messages = [{"role": "system", "content": system_prompt}]
        
        while True:
            # Entrada do usuário
            user_input = input("\nVocê: ").strip()
            
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("\nEncerrando chat...")
                break
            
            if not user_input:
                continue
            
            # Adicionar mensagem do usuário
            messages.append({"role": "user", "content": user_input})
            
            # Formatar histórico para o modelo
            prompt = self._format_chat_prompt(messages)
            
            print("\nMistral: ", end="", flush=True)
            
            try:
                # Gerar resposta com streaming
                response = self.llm(
                    prompt,
                    max_tokens=512,
                    temperature=0.7,
                    top_p=0.95,
                    echo=False,
                    stop=["[INST]", "[/INST]", "\n\n"],
                    stream=True
                )
                
                # Exibir resposta em tempo real
                full_response = ""
                for chunk in response:
                    text = chunk['choices'][0]['text']
                    print(text, end="", flush=True)
                    full_response += text
                
                # Adicionar resposta ao histórico
                messages.append({"role": "assistant", "content": full_response.strip()})
                
            except Exception as e:
                print(f"\nErro: {str(e)}")
    
    def _format_chat_prompt(self, messages):
        """
        Formata mensagens para o modelo Mistral
        """
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"[INST] {message['content']} [/INST]\n\n"
            elif message["role"] == "user":
                prompt += f"[INST] {message['content']} [/INST]\n\n"
            elif message["role"] == "assistant":
                prompt += f"{message['content']}\n\n"
        
        return prompt.strip()

def main():
    # Caminho do modelo
    MODEL_PATH = "models/mistral-7b-instruct.gguf"
    
    # Inicializar LLM
    print("🚀 Inicializando LLM Local...")
    llm = LocalLLM(MODEL_PATH)
    
    # Menu de opções
    while True:
        print("\n" + "="*50)
        print("LLM Local - Mistral 7B")
        print("="*50)
        print("1. Gerar texto")
        print("2. Modo chat")
        print("3. Teste rápido")
        print("4. Sair")
        print("="*50)
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == "1":
            prompt = input("\nDigite seu prompt: ")
            response = llm.generate(prompt)
            print(f"\nResposta:\n{response}")
            
        elif choice == "2":
            system_msg = input("Mensagem do sistema (ou Enter para padrão): ")
            if not system_msg:
                system_msg = "Você é um assistente útil."
            llm.chat(system_msg)
            
        elif choice == "3":
            # Testes rápidos
            tests = [
                "Explique o que é machine learning em 3 frases.",
                "Escreva um poema sobre tecnologia.",
                "Quais são as vantagens de usar Python?"
            ]
            
            for test in tests:
                print(f"\nPrompt: {test}")
                response = llm.generate(test, max_tokens=100)
                print(f"Resposta: {response}")
                print("-"*50)
                
        elif choice == "4":
            print("\n👋 Encerrando...")
            break
            
        else:
            print("\n❌ Opção inválida!")

if __name__ == "__main__":
    main()




# 11. Criar codigo Python para configurações 
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



# 12. Criar codigo do Chat 
# run.py
#!/usr/bin/env python3
"""
Script principal para rodar a LLM local
"""
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from main import main

if __name__ == "__main__":
    print("="*60)
    print("🤖 LLM Local - Mistral 7B")
    print("="*60)
    
    # Verificar se o modelo existe
    model_path = "models/mistral-7b-instruct.gguf"
    if not os.path.exists(model_path):
        print(f"\n❌ ERRO: Modelo não encontrado!")
        print(f"Por favor, coloque o modelo em: {model_path}")
        print("\nDica: Você já baixou o modelo?")
        print("Comando para verificar: ls models/")
        sys.exit(1)
    
    # Rodar main
    main()


PRONTO python run.py
op 2

# Daqui para baixo é apenas para testes...
# 13. Verificar estrutura
tree llm_local/

# 14. Rodar o script
python run.py

# 15. Teste rápido no terminal
python -c "
from llama_cpp import Llama
import sys

model_path = 'models/mistral-7b-instruct.gguf'
print(f'Testando modelo: {model_path}')

llm = Llama(model_path=model_path, n_ctx=512)
response = llm('Olá, como você está?', max_tokens=50)
print('Resposta:', response['choices'][0]['text'])
"



# 16. otimização
# src/optimize.py
"""
Otimizações para melhor performance
"""

class OptimizedLLM:
    def __init__(self, model_path: str):
        # Verificar hardware disponível
        self.use_gpu = self._check_gpu_support()
        
        # Configurar parâmetros baseado no hardware
        if self.use_gpu:
            print("⚡ Usando GPU para aceleração")
            n_gpu_layers = -1  # Usar todas as camadas na GPU
        else:
            print("💻 Usando CPU apenas")
            n_gpu_layers = 0
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,  # Contexto maior se tiver memória
            n_threads=self._get_optimal_threads(),
            n_gpu_layers=n_gpu_layers,
            n_batch=512,  # Tamanho do batch
            verbose=False
        )
    
    def _check_gpu_support(self):
        """Verifica suporte a GPU"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _get_optimal_threads(self):
        """Calcula número ótimo de threads"""
        import multiprocessing
        cores = multiprocessing.cpu_count()
        # Usar 75% dos cores disponíveis
        return max(1, int(cores * 0.75))
