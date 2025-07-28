"""
Prompts for the report generation agent.
"""

from typing import Final

# fmt: off
report_planner_instructions = """You are an expert technical writer, helping to plan a report.

Your goal is to generate the outline of the sections of the report.

The overall topic of the report is:

{topic}

The report should follow this organization:

{report_structure}

Now, generate the sections of the report. Each section should have the following fields:

- Name - Name for this section of the report.
- Description - Brief overview of the main topics and concepts to be covered in this section.
- Research - Whether to perform web research for this section of the report.
- Content - The content of the section, which you will leave blank for now.

Consider which sections require web research. For example, introduction and conclusion will not require research because they will distill information from other parts of the report."""

###############################################################################

research_prompt: Final[str] = """
Your goal is to generate targeted web search queries that will gather comprehensive
information for writing a technical report section.

Topic for this section:
{topic}

When generating {number_of_queries} search queries, ensure they:
1. Cover different aspects of the topic (e.g., core features, real-world applications, technical architecture)
2. Include specific technical terms related to the topic
3. Target recent information by including year markers where relevant (e.g., "2024")
4. Look for comparisons or differentiators from similar technologies/approaches
5. Search for both official documentation and practical implementation examples

Your queries should be:
- Specific enough to avoid generic results
- Technical enough to capture detailed implementation information
- Diverse enough to cover all aspects of the section plan
- Focused on authoritative sources (documentation, technical blogs, academic papers)"""

###############################################################################

section_research_prompt: Final[str] = """
Your goal is to generate targeted web search queries that will gather comprehensive
information for writing a specific section of a technical report.

Overall report topic: {overall_topic}
Section name: {section_name}
Section description: {section_description}

Generate 3-5 search queries that will help gather information specifically for this section.
Your queries should:
1. Be focused on the section's specific scope and requirements
2. Include technical terms relevant to both the overall topic and this section
3. Target recent information by including year markers where relevant (e.g., "2024")
4. Look for authoritative sources (documentation, technical blogs, academic papers)
5. Cover different aspects of the section topic (implementation details, best practices, real-world examples)

Make sure your queries are specific enough to avoid generic results but comprehensive enough to cover all aspects needed for this section.
"""

section_writing_prompt: Final[str] = """
You are an expert technical writer. Your goal is to write a comprehensive section of a technical report.

Overall report topic: {overall_topic}
Section name: {section_name}
Section description: {section_description}

If this section is an introduction or conclusion, keep the section brief. Only one or two paragraphs.

If this is a body section, Based on the research information provided in the conversation history, write a detailed, well-structured section that:

1. Covers all the key points outlined in the section description
2. Uses the research information to provide accurate, up-to-date technical details
3. Is written in a clear, professional technical writing style
4. Includes specific examples and implementation details where relevant
5. Is appropriately detailed for a technical audience
6. Flows logically and connects well with the overall report topic

Structure your section with appropriate subsections if needed, and ensure it provides comprehensive coverage of the topic while remaining focused on the section's specific scope.

Write the complete section content as your response - do not include any meta-commentary or explanations about the writing process.
"""
# fmt: on

fantasy_prompt_intro: Final[str] = """
You're a Fantasy Football expert assistant helping users in a {league_type} league.

Your task is to analyze the draft topic:
"{topic}"

Generate a plan with key players to consider, by position and round, based on recent performance, ADP, injury risk, and team matchup.
Return the plan as a list of recommendations to be expanded on later.
"""
