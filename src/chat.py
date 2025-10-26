from search import search_prompt

def main():
    """
    Função principal que inicia o chat CLI interativo.
    """
    print("=" * 60)
    print("🤖 CHAT DE BUSCA SEMÂNTICA")
    print("=" * 60)
    print("Digite 'sair' para encerrar o chat.")
    print("=" * 60)
    
    search_chain = search_prompt()

    if not search_chain:
        print("❌ Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        print("Certifique-se de que:")
        print("1. O banco de dados PostgreSQL está rodando (docker compose up -d)")
        print("2. O PDF foi ingerido (python src/ingest.py)")
        print("3. As variáveis de ambiente estão configuradas corretamente")
        return
    
    print("✅ Chat inicializado com sucesso!")
    print()
    
    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
            
            if pergunta.lower() in ['sair', 'quit', 'exit', 'q']:
                print("👋 Encerrando o chat. Até logo!")
                break
            
            if not pergunta:
                print("⚠️  Por favor, digite uma pergunta válida.")
                continue
            
            print("🔍 Buscando informações...")
            
            resposta = search_chain["search_and_answer"](pergunta)
            
            print("RESPOSTA:", resposta)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Chat encerrado pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")
            print("Tente novamente ou digite 'sair' para encerrar.")

if __name__ == "__main__":
    main()