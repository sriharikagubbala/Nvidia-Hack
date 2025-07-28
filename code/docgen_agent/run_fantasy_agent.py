from docgen_agent.fantasy_agent import FantasyAgentState, graph

def main():
    state = FantasyAgentState(
        topic="Round 4 Wide Receivers",
        league_type="PPR"
    )

    output = graph.invoke(state)

    print("\n=== Fantasy Draft Recommendation ===\n")
    print(output.recommendation_output or output.get("recommendation_output"))

if __name__ == "__main__":
    main()
