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

