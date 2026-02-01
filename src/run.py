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


