# llm_chat.py - 30 linhas
from llama_cpp import Llama

# 1-3: Configurar
model = Llama(
    model_path="models/mistral-7b-instruct.gguf",
    n_ctx=2048,
    verbose=False
)

# 4-15: Função de chat
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

# 16-17: Rodar
if __name__ == "__main__":
    chat()