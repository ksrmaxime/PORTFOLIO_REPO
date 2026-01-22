from portfolio_repo.llm import get_local_llm

if __name__ == "__main__":
    llm = get_local_llm()

    out = llm.chat(
        messages=[
            {"role": "system", "content": "You are a precise classification assistant."},
            {"role": "user", "content": "Answer only: YES or NO. Is this text about AI regulation?"},
        ],
        temperature=0.0,
        max_tokens=5,
    )

    print(out)
