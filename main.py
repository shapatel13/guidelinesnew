import streamlit as st
import datetime
import json
import os
import re
import logging
from textwrap import dedent
import time

from agno.agent import Agent
from agno.models.perplexity import Perplexity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('guideline_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Use perplexity API key from environment
perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
if not perplexity_api_key:
    logger.error("PERPLEXITY_API_KEY not found in environment variables")

# Initialize the agent with Perplexity model
def create_perplexity_agent(use_fast_research=False):
    model_id = "sonar-pro" if use_fast_research else "sonar-deep-research"
    return Agent(
        model=Perplexity(id=model_id, api_key=perplexity_api_key),
        description="Expert medical guideline researcher and analyst",
        markdown=True
    )

# Function to research guideline metadata
def research_guideline_metadata(topic, use_fast_research=False, use_cache=True):
    """Research guideline metadata and executive summary"""
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create cache if needed
    cache_file = "tmp/guidelines_cache.json"
    cache_key = f"{topic}_metadata_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Add research mode to cache key to avoid mixing fast and deep research results
    research_mode = "fast" if use_fast_research else "deep"
    cache_key = f"{cache_key}_{research_mode}"

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache_key in cache:
                    logger.info("Using cached metadata")
                    return cache[cache_key]
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")

    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Prepare metadata prompt
    prompt = f"""
# Guideline Research Task: {topic} - Metadata and Executive Summary

## Background Context
Medical guidelines require regular updates as new evidence emerges. You are tasked with creating an updated medical guideline framework for {topic} based on the most recent authoritative guidelines and new evidence published since then.

## Task Overview
1. Identify the most recent authoritative {topic} guidelines from major medical societies (ATS, IDSA, BTS, etc.)
2. Create a comprehensive executive summary and references framework for an updated guideline

## Research Instructions - IMPORTANT
1. First identify and analyze:
    - The most recent authoritative {topic} guidelines (specific document name)
    - The publishing organization(s)
    - The exact publication date (month/year)
    - The primary authors/committee
    - The evidence grading methodology used (e.g., GRADE, Oxford, proprietary system)

2. Write a comprehensive executive summary (4-5 paragraphs) that:
    - Explains the scope and significance of {topic} as a clinical issue
    - Identifies 5-7 major areas where significant new evidence has emerged since the original guidelines
    - Highlights specific studies or meta-analyses with precise statistics that have changed practice
    - Explains potential impacts of these changes on patient outcomes with quantitative measures
    - Discusses any controversies or areas of continued debate in the field

3. Research and create a comprehensive references framework with:
    - 5-10 citations to the original guidelines and related foundation documents
    - 15-20 high-impact studies published AFTER the original guidelines
    - Clear explanation of why each new study is practice-changing
    - Specific outcome measures and statistical significance from these studies

## Output Format
Format your response as follows:

# {topic} Guidelines Update (2025)

## Original Publication: [Month YEAR] by [ORGANIZATION] | Current Update: April 2025

### Executive Summary
[4-5 detailed paragraphs with specific study mentions and statistics]

## Evidence Framework and Key Updates
1. [Area of practice with significant changes]
    - Original guideline approach: [Brief description]
    - New evidence: [Specific studies with statistics]
    - Clinical implication: [How practice should change]

2. [Second area with changes]
    [Follow same format]

[Continue for all major areas]

## References
1. [Original guideline citation]
2-5. [Other foundation documents]
6-25. [New evidence citations in standard academic format]
"""

    try:
        # Run the agent
        logger.info(f"Researching metadata for {topic} using {'fast' if use_fast_research else 'deep'} research...")
        response = agent.run(prompt)

        # Process response
        if response and response.content:
            result = str(response.content)
            logger.info(f"Generated metadata of length: {len(result)}")

            # Cache the result
            if use_cache:
                try:
                    cache = {}
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            cache = json.load(f)
                    cache[cache_key] = result
                    with open(cache_file, "w") as f:
                        json.dump(cache, f)
                except Exception as e:
                    logger.error(f"Error writing to cache: {str(e)}")

            return result
        else:
            logger.error(f"No metadata received")
            return "# Guidelines Update\n\n## Failed to generate metadata.\n\nPlease check the logs for details."
    except Exception as e:
        logger.error(f"Error researching metadata: {str(e)}")
        return f"# Guidelines Update\n\n## Error\n\nAn error occurred while researching metadata: {str(e)}"

# Function to research guideline section
def research_guideline_section(topic, section, use_fast_research=False, use_cache=True):
    """Research a specific section of the guidelines"""
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create cache if needed
    cache_file = "tmp/guidelines_cache.json"
    cache_key = f"{topic}_{section}_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Add research mode to cache key
    research_mode = "fast" if use_fast_research else "deep"
    cache_key = f"{cache_key}_{research_mode}"

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache_key in cache:
                    logger.info(f"Using cached result for {section}")
                    return cache[cache_key]
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")

    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Section-specific prompt
    prompt = f"""
# Guideline Research Task: {topic} - {section} Section

## Research Context
You are a medical guideline development expert specializing in {topic}. Your task is to conduct original analysis comparing existing {section} recommendations with the latest evidence to create evidence-based updates.

## Critical Research Objectives
1. First identify:
    - Exactly what the most recent authoritative guidelines recommend for {section}
    - The evidence grade assigned to each recommendation
    - Limitations explicitly acknowledged in the original guideline

2. Then thoroughly research:
    - Studies published AFTER the original guidelines specifically about {section}
    - Focus on Level 1 evidence: systematic reviews, meta-analyses, large multi-center RCTs
    - Identify studies with statistical significance that challenge or strengthen original recommendations
    - Look for evidence gaps that have been addressed since the original publication

3. For each recommendation, provide:
    - Original recommendation with exact wording and evidence grade
    - Comprehensive analysis of new evidence (with specific p-values, confidence intervals, etc.)
    - Clearly articulated rationale for modifying or retaining the recommendation
    - Specific implementation considerations for the updated recommendation

## Synthesis Requirements
1. Perform original analysis:
    - Don't simply summarize new studies - analyze how they collectively affect clinical practice
    - Identify patterns across multiple studies that suggest a clear direction for practice
    - Articulate practical implications for clinicians implementing these recommendations
    - Address how the recommendation applies to different patient subgroups

2. Create side-by-side comparison:
    - Original recommendation (exact wording) vs. proposed updated recommendation
    - Bold ALL changes in the updated version with explanations for each change
    - Provide precise rationale linked to specific evidence for every modification
    - Assign appropriate evidence grades using the same methodology as the original guideline

3. For controversial areas:
    - Acknowledge ongoing debates with balanced presentation of evidence
    - Provide clinical guidance despite uncertainty when appropriate
    - Suggest practical approaches while research continues

## Output Format
Use this exact format:

### {section}

| Original Recommendation (YEAR) | Updated Recommendation (2025) |
|--------------------------------|--------------------------------|
| Original text [Grade X] | Updated text with **bold changes** [Grade Y] |
| Second recommendation [Grade X] | Updated recommendation with **bold changes** [Grade Y] |

#### Rationale for Changes
1. **[Key change #1]**:
    - Evidence: [Specific studies with detailed statistics]
    - Analysis: [Interpretation of how this evidence impacts practice]
    - Implementation: [Practical considerations for clinicians]

2. **[Key change #2]**:
    [Same detailed format]

#### Special Considerations
- [Patient subgroups]
- [Implementation challenges]
- [Resource implications]
"""

    try:
        # Run the agent
        logger.info(f"Researching {section} for {topic} using {'fast' if use_fast_research else 'deep'} research...")
        response = agent.run(prompt)

        # Check response
        if response and response.content:
            result = str(response.content)
            logger.info(f"Generated content for {section} of length: {len(result)}")

            # Cache the result
            if use_cache:
                try:
                    cache = {}
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            cache = json.load(f)
                    cache[cache_key] = result
                    with open(cache_file, "w") as f:
                        json.dump(cache, f)
                except Exception as e:
                    logger.error(f"Error writing to cache: {str(e)}")

            return result
        else:
            logger.error(f"No content received for {section}")
            return f"### {section}\n\nFailed to generate content for this section. Please check the logs for details."
    except Exception as e:
        logger.error(f"Error researching {section}: {str(e)}")
        return f"### {section}\n\nAn error occurred while researching this section: {str(e)}"

# Function for research new recommendations
def research_new_recommendations(topic, use_fast_research=False, use_cache=True):
    """Research completely new recommendations"""
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create cache if needed
    cache_file = "tmp/guidelines_cache.json"
    cache_key = f"{topic}_new_recommendations_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Add research mode to cache key
    research_mode = "fast" if use_fast_research else "deep"
    cache_key = f"{cache_key}_{research_mode}"

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache_key in cache:
                    logger.info(f"Using cached new recommendations")
                    return cache[cache_key]
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")

    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Prepare new recommendations prompt
    prompt = f"""
# Guideline Research Task: {topic} - Completely New Recommendations

## Research Context
You are developing entirely new recommendations for {topic} guidelines that were NOT addressed in previous guidelines. These should reflect emerging evidence, new technologies, or evolving clinical practices.

## Research Objectives
1. First identify critical gaps in current guidelines:
    - Areas where clinicians lack guidance but make frequent decisions
    - New technologies or approaches developed since the original guidelines
    - Patient populations or scenarios not addressed in original guidelines
    - Areas where practice has evolved significantly without formal guidance

2. Then conduct comprehensive research on:
    - High-quality evidence supporting these new clinical approaches
    - Consensus statements from expert groups on these topics
    - Emerging standards of care that have developed organically
    - International variations in approach to these scenarios

3. For each new recommendation:
    - Develop precise, actionable clinical guidance
    - Ground each recommendation in specific studies (with statistics)
    - Assign appropriate evidence grades using the original guideline methodology
    - Provide detailed implementation guidance
    - Address potential barriers, costs, and training needs

## Requirements for Each New Recommendation
1. Clinical relevance:
    - Address a frequent clinical scenario or decision point
    - Provide clear guidance that changes clinical behavior
    - Focus on patient-important outcomes

2. Evidence quality:
    - Base recommendations on the highest quality available evidence
    - Acknowledge limitations of evidence transparently
    - Provide balanced interpretation of controversial evidence

3. Practical implementation:
    - Create recommendations that are feasible in various settings
    - Consider resource implications and cost-effectiveness
    - Address training needs or system changes required
    - Provide guidance for different resource environments

## Output Format
Use this exact format:

## New Recommendations

### [Category 1 - e.g., Prevention, Diagnosis, etc.]

1. **"[New recommendation exact text]"** [Suggested Grade: X]
    - **Rationale**: [Detailed explanation with at least 3 supporting studies including sample sizes and statistics]
    - **Evidence Quality**: [Analysis of evidence strength and limitations]
    - **Implementation Considerations**: [Practical guidance for clinicians]
    - **Resource Implications**: [Consideration of costs, equipment, training]
    - **Special Populations**: [Modifications needed for specific patient groups]

### [Category 2 - Different aspect]

2. **"[New recommendation exact text]"** [Suggested Grade: X]
    [Follow same comprehensive format]

[Continue for 4-6 total new recommendations]
"""

    try:
        # Run the agent
        logger.info(f"Researching new recommendations for {topic} using {'fast' if use_fast_research else 'deep'} research...")
        response = agent.run(prompt)

        # Process response
        if response and response.content:
            result = str(response.content)
            logger.info(f"Generated new recommendations of length: {len(result)}")

            # Cache the result
            if use_cache:
                try:
                    cache = {}
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            cache = json.load(f)
                    cache[cache_key] = result
                    with open(cache_file, "w") as f:
                        json.dump(cache, f)
                except Exception as e:
                    logger.error(f"Error writing to cache: {str(e)}")

            return result
        else:
            logger.error(f"No new recommendations received")
            return "## New Recommendations\n\nFailed to generate new recommendations. Please check the logs for details."
    except Exception as e:
        logger.error(f"Error researching new recommendations: {str(e)}")
        return f"## New Recommendations\n\nAn error occurred while researching new recommendations: {str(e)}"

# Function for chunked section research
def research_section_chunked(topic, section, use_fast_research=False, use_cache=True):
    """Research a section in chunks to prevent truncation"""
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create cache if needed
    cache_file = "tmp/guidelines_cache.json"
    cache_key = f"{topic}_{section}_chunked_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Add research mode to cache key
    research_mode = "fast" if use_fast_research else "deep"
    cache_key = f"{cache_key}_{research_mode}"

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache_key in cache:
                    logger.info(f"Using cached chunked result for {section}")
                    return cache[cache_key]
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")

    # Step 1: First get the original guideline recommendations for this section
    original_recommendations = research_original_recommendations(topic, section, use_fast_research)

    # Step 2: Research new evidence for each recommendation
    evidence_analysis = research_section_evidence(topic, section, original_recommendations, use_fast_research)

    # Step 3: Generate the updated recommendations with rationale
    updated_recommendations = generate_updated_recommendations(topic, section, original_recommendations, evidence_analysis, use_fast_research)

    # Combine all parts
    full_section = f"""
### {section}

{updated_recommendations}
    """

    # Cache the result
    if use_cache:
        try:
            cache = {}
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            cache[cache_key] = full_section
            with open(cache_file, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")

    return full_section

def research_original_recommendations(topic, section, use_fast_research=False):
    """Research just the original recommendations for a section"""
    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Focused prompt to get just the original recommendations
    prompt = f"""
# Research Task: Original {topic} Guidelines for {section}

## Task Description
Research and identify the exact recommendations from the most authoritative and recent guidelines on {topic}, focusing ONLY on the {section} aspect.

## Research Instructions
1. Identify the most recent/authoritative guidelines for {topic}
2. Extract ONLY the specific recommendations related to {section}
3. Include the exact wording, evidence grade, and year published
4. Be comprehensive - include ALL recommendations in this section

## Output Format
- Format as a numbered list
- For each recommendation include:
  1. The exact text of the recommendation
  2. The evidence grade (A, B, C, etc. or equivalent)
  3. The year published

Example:
1. "Recommendation text exactly as written in guidelines" [Grade B, 2019]
2. "Second recommendation text" [Grade A, 2019]
"""

    # Run the agent
    logger.info(f"Researching original recommendations for {section} using {'fast' if use_fast_research else 'deep'} research...")
    response = agent.run(prompt)

    # Process response
    if response and response.content:
        result = str(response.content)
        logger.info(f"Retrieved original recommendations: {len(result)} chars")
        return result
    else:
        logger.error(f"Failed to retrieve original recommendations for {section}")
        return f"1. \"No specific recommendations found for {section}\" [Grade Unknown, Unknown date]"

def research_section_evidence(topic, section, original_recommendations, use_fast_research=False):
    """Research new evidence for recommendations"""
    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Focused prompt for evidence analysis
    prompt = f"""
# Research Task: New Evidence Analysis for {topic} - {section}

## Original Recommendations
{original_recommendations}

## Task Description
Research recent evidence (published AFTER the original guidelines) that would impact each of the recommendations above. Focus on:
- Systematic reviews
- Meta-analyses
- Large randomized controlled trials
- New clinical guidelines from other organizations

## Research Instructions
For EACH original recommendation:
1. Find 2-3 high-quality studies published AFTER the original guideline that:
    - Support, contradict, or refine the recommendation
    - Provide new statistical data (include p-values, confidence intervals, etc.)
    - Have methodological rigor

2. Analyze how this evidence would impact the recommendation:
    - Would it strengthen or weaken the recommendation?
    - Would it change the evidence grade?
    - Would it modify specific elements of the recommendation?
    - Would it expand the recommendation to new populations?

## Output Format
Format your analysis for each recommendation:

### Evidence for Recommendation 1
- **Study 1**: [Authors, Title, Journal Year] - Sample size, design, key findings with statistics
  - Impact: How this would change the recommendation
- **Study 2**: [Citation] - Details and impact

### Evidence for Recommendation 2
[Same format]
"""

    # Run the agent
    logger.info(f"Researching new evidence for {section} using {'fast' if use_fast_research else 'deep'} research...")
    response = agent.run(prompt)

    # Process response
    if response and response.content:
        result = str(response.content)
        logger.info(f"Retrieved evidence analysis: {len(result)} chars")
        return result
    else:
        logger.error(f"Failed to retrieve evidence analysis for {section}")
        return f"### Evidence Analysis\nNo substantial new evidence was found that would change the original recommendations."

def generate_updated_recommendations(topic, section, original_recommendations, evidence_analysis, use_fast_research=False):
    """Generate updated recommendations based on original recs and new evidence"""
    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Focused prompt for final recommendation updates
    prompt = f"""
# Task: Generate Updated Recommendations for {topic} - {section}

## Original Recommendations
{original_recommendations}

## Evidence Analysis
{evidence_analysis}

## Task Description
Based on the original recommendations and new evidence analysis, create a side-by-side comparison of original vs. updated recommendations with detailed rationale.

## Instructions for Updates
1. Create a clear comparison table showing original and updated recommendations
2. BOLD all changes in the updated recommendations
3. For each change:
    - Provide specific rationale tied to evidence
    - Update the evidence grade if warranted
    - Be specific and actionable in the recommendation language

4. Exercise original thinking:
    - Don't just make minor wording changes - consider real clinical implications
    - Evaluate if recommendations should be strengthened, weakened, or qualified
    - Consider how implementation would work in clinical practice
    - Address any controversies or debates in the field

## Output Format

| Original Recommendation | Updated Recommendation |
|-------------------------|------------------------|
| Original text [Grade X] | Updated text with **bold changes** [Grade Y] |

#### Rationale for Changes
1. **First change**: Evidence and reasoning
2. **Second change**: Evidence and reasoning

[Repeat for each recommendation]
"""

    # Run the agent
    logger.info(f"Generating updated recommendations for {section} using {'fast' if use_fast_research else 'deep'} research...")
    response = agent.run(prompt)

    # Process response
    if response and response.content:
        result = str(response.content)
        logger.info(f"Generated updated recommendations: {len(result)} chars")
        return result
    else:
        logger.error(f"Failed to generate updated recommendations for {section}")
        return f"No updates to {section} recommendations could be generated based on current evidence."

# Function for chunked new recommendations
def research_new_recommendations_chunked(topic, use_fast_research=False, use_cache=True):
    """Research completely new recommendations in chunks"""
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create cache if needed
    cache_file = "tmp/guidelines_cache.json"
    cache_key = f"{topic}_new_recs_chunked_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Add research mode to cache key
    research_mode = "fast" if use_fast_research else "deep"
    cache_key = f"{cache_key}_{research_mode}"

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache_key in cache:
                    logger.info(f"Using cached chunked new recommendations")
                    return cache[cache_key]
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")

    # Step 1: Identify gaps in current guidelines
    guideline_gaps = identify_guideline_gaps(topic, use_fast_research)

    # Step 2: Research each gap area
    gap_analyses = []

    # Extract gap areas from the response
    gap_areas = extract_gap_areas(guideline_gaps)

    # Research each gap area
    for gap in gap_areas:
        gap_analysis = research_single_gap(topic, gap, use_fast_research)
        gap_analyses.append(gap_analysis)

    # Step 3: Compile all new recommendations
    all_new_recommendations = "\n\n".join(gap_analyses)

    # Format the final output
    new_recommendations = f"""
## New Recommendations

{all_new_recommendations}
    """

    # Cache the result
    if use_cache:
        try:
            cache = {}
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    cache = json.load(f)
            cache[cache_key] = new_recommendations
            with open(cache_file, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            logger.error(f"Error writing to cache: {str(e)}")

    return new_recommendations

def identify_guideline_gaps(topic, use_fast_research=False):
    """Identify gaps in current guidelines that need new recommendations"""
    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Focused prompt for gap identification
    prompt = f"""
# Research Task: Identify Gaps in Current {topic} Guidelines

## Task Description
Identify 3-5 specific clinical areas related to {topic} that are NOT adequately addressed in current guidelines but where clinicians need guidance.

## Research Instructions
1. Research the most recent authoritative guidelines on {topic}
2. Identify areas where:
    - No specific recommendations exist
    - Evidence has emerged since guideline publication
    - Clinical practice has evolved significantly
    - New technologies or approaches have been developed
    - Special populations aren't adequately addressed

## Output Format
List 3-5 specific gap areas in this format:

### Gap Areas in Current Guidelines

1. **[Gap Area 1]**
    - Why this is a gap: [Explanation]
    - Clinical importance: [Why clinicians need guidance here]
    - Types of emerging evidence: [What new research exists]

2. **[Gap Area 2]**
    [Same format]

[Continue for all identified gaps]
"""

    # Run the agent
    logger.info(f"Identifying gaps in current {topic} guidelines using {'fast' if use_fast_research else 'deep'} research...")
    response = agent.run(prompt)

    # Process response
    if response and response.content:
        result = str(response.content)
        logger.info(f"Identified guideline gaps: {len(result)} chars")
        return result
    else:
        logger.error(f"Failed to identify guideline gaps for {topic}")
        return "### Gap Areas in Current Guidelines\n\n1. **Implementation Guidance**\n   - Why this is a gap: Current guidelines lack specific implementation strategies\n   - Clinical importance: Clinicians need practical guidance for real-world application"

def extract_gap_areas(gap_analysis):
    """Extract gap areas from the gap analysis text"""
    # Simple extraction using regex
    # Look for markdown headings or bold text patterns
    gap_areas = []

    # Try to find bold text pattern first
    bold_pattern = r'\*\*([^*]+)\*\*'
    matches = re.findall(bold_pattern, gap_analysis)

    if matches:
        # Filter out any that don't look like gap areas (too short or common headers)
        gap_areas = [match for match in matches if len(match) > 5 and "Gap Area" not in match]

    # If we couldn't find any, or found too few, return default areas
    if len(gap_areas) < 2:
        gap_areas = [
            "Implementation Strategies",
            "Special Populations",
            "New Technologies"
        ]

    return gap_areas[:5]  # Limit to 5 areas

def research_single_gap(topic, gap_area, use_fast_research=False):
    """Research a single gap area to develop new recommendations"""
    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Focused prompt for single gap research
    prompt = f"""
# Research Task: Develop New {topic} Recommendations for {gap_area}

## Task Description
Create 1-2 new evidence-based recommendations for {topic} addressing the gap area of {gap_area}.

## Research Instructions
1. Research recent high-quality evidence related to {gap_area} in {topic}
2. Identify specific clinical questions that need guidance
3. Develop precise, actionable recommendations that:
    - Address a specific clinical scenario
    - Are based on best available evidence
    - Include appropriate evidence grading
    - Provide implementation guidance

## Output Format
Format your new recommendations as:

### {gap_area}

1. **"[Exact recommendation text]"** [Suggested Grade: X]
    - **Rationale**: [Evidence-based justification with 2-3 key studies]
    - **Implementation**: [Practical guidance for clinicians]
    - **Special Considerations**: [Important caveats or subpopulations]

2. **"[Second recommendation if applicable]"** [Suggested Grade: X]
    [Same format as above]
"""

    # Run the agent
    logger.info(f"Researching new recommendations for {gap_area} using {'fast' if use_fast_research else 'deep'} research...")
    response = agent.run(prompt)

    # Process response
    if response and response.content:
        result = str(response.content)
        logger.info(f"Generated new recommendations for {gap_area}: {len(result)} chars")
        return result
    else:
        logger.error(f"Failed to generate new recommendations for {gap_area}")
        return f"### {gap_area}\n\nInsufficient evidence is currently available to make formal recommendations in this area."

def extract_key_points(metadata_result, sections_summary):
    """Extract key points from previous results to ground the conclusion"""
    # Simple extraction for demo purposes
    key_points = "Key points from previous sections:\n"

    # Extract publication date and org if available
    pub_date_match = re.search(r'Original Publication: (\w+ \d{4}) by ([^|]+)', metadata_result)
    if pub_date_match:
        key_points += f"- Original guidelines published {pub_date_match.group(1)} by {pub_date_match.group(2).strip()}\n"

    # Extract section names
    section_matches = re.findall(r'### ([^\n]+)', sections_summary)
    if section_matches:
        key_points += "- Updated sections include: " + ", ".join(section_matches) + "\n"

    # Extract boldface changes
    bold_changes = re.findall(r'\*\*([^*]+)\*\*', sections_summary)
    if bold_changes and len(bold_changes) <= 10:
        key_points += "- Key changes include: " + ", ".join(bold_changes[:5]) + "\n"
    elif bold_changes:
        key_points += f"- Approximately {len(bold_changes)} significant changes have been made across all sections\n"

    return key_points

# Function for comprehensive conclusion
def research_comprehensive_conclusion(topic, metadata_result, sections_summary, use_fast_research=False, use_cache=True):
    """Create a comprehensive conclusion to ensure completeness"""
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create cache if needed
    cache_file = "tmp/guidelines_cache.json"
    cache_key = f"{topic}_conclusion_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Add research mode to cache key
    research_mode = "fast" if use_fast_research else "deep"
    cache_key = f"{cache_key}_{research_mode}"

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache_key in cache:
                    logger.info("Using cached conclusion")
                    return cache[cache_key]
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")

    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Extract key points to ground the conclusion
    key_points = extract_key_points(metadata_result, sections_summary)

    # Focused prompt for conclusion
    prompt = f"""
# Task: Create Comprehensive Conclusion for {topic} Guidelines

## Key Points from Previous Sections
{key_points}

## Task Description
Create a comprehensive conclusion section that integrates all guideline updates and provides implementation guidance. This conclusion should ensure the document feels complete.

## Required Elements
1. Summary of Key Changes:
    - Highlight the 5-7 most significant changes to previous guidelines
    - Emphasize the strength of evidence behind these changes
    - Explain how these changes will impact patient outcomes

2. Implementation Strategy:
    - Provide a phased approach to implementing these updates
    - Address potential barriers and how to overcome them
    - Suggest quality metrics to monitor implementation

3. Future Research Priorities:
    - Identify 3-5 specific research questions that need to be addressed
    - Explain how answering these questions would advance care
    - Suggest study designs that would best address these gaps
    
4. Review Schedule:
    - Recommend when these guidelines should next be reviewed
    - Identify triggers that would necessitate earlier review
    - Suggest monitoring systems for emerging evidence

## Output Format
Format your response as:

## Conclusion and Implementation

### Summary of Key Changes
[Comprehensive summary of major updates]

### Implementation Strategy
[Detailed, practical implementation guidance]

### Future Research Priorities
[Specific research questions and study designs]

### Review Schedule
[Recommendations for ongoing monitoring and updates]

---

## Appendices

### Appendix A: Evidence Grading Methodology
[Explanation of grading system used]

### Appendix B: Guideline Development Process
[Brief explanation of how this update was created]

### Appendix C: Conflicts of Interest
[Standard statement about conflicts of interest]
"""

    try:
        # Run the agent
        logger.info(f"Creating comprehensive conclusion for {topic} using {'fast' if use_fast_research else 'deep'} research...")
        response = agent.run(prompt)

        # Process response
        if response and response.content:
            result = str(response.content)
            logger.info(f"Generated conclusion: {len(result)} chars")

            # Cache the result
            if use_cache:
                try:
                    cache = {}
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            cache = json.load(f)
                    cache[cache_key] = result
                    with open(cache_file, "w") as f:
                        json.dump(cache, f)
                except Exception as e:
                    logger.error(f"Error writing to cache: {str(e)}")

            return result
        else:
            logger.error(f"Failed to generate conclusion for {topic}")
            return "## Conclusion\n\nThis concludes the updated guidelines. Implementation should be tailored to local contexts and resources."
    except Exception as e:
        logger.error(f"Error researching conclusion: {str(e)}")
        return f"## Conclusion\n\nAn error occurred while researching the conclusion: {str(e)}"

# Function to adapt guidelines for different contexts
def generate_context_adaptations(topic, sections_content, use_fast_research=False, use_cache=True):
    """Generate context-specific adaptations for different healthcare settings"""
    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Create cache if needed
    cache_file = "tmp/guidelines_cache.json"
    cache_key = f"{topic}_context_adaptations_{datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    # Add research mode to cache key
    research_mode = "fast" if use_fast_research else "deep"
    cache_key = f"{cache_key}_{research_mode}"

    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache_key in cache:
                    logger.info("Using cached context adaptations")
                    return cache[cache_key]
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")

    # Create a new agent
    agent = create_perplexity_agent(use_fast_research)

    # Extract recommendations to adapt
    recommendations = extract_recommendations(sections_content)

    # Prompt for context adaptations
    prompt = f"""
# Task: Generate Setting-Specific Adaptations for {topic} Guidelines

## Guidelines Overview
{recommendations}

## Task Description
Create practical adaptations of these guidelines for three different healthcare settings:
1. Resource-Limited Settings (e.g., rural clinics, low-income areas)
2. Primary Care Settings (e.g., outpatient clinics, community health centers)
3. Specialty Care Settings (e.g., tertiary hospitals, academic medical centers)

## Required Elements for Each Setting
For each healthcare setting, provide:

1. Resource Considerations:
   - Equipment and technology requirements
   - Staffing implications
   - Cost considerations
   - Accessibility challenges

2. Implementation Priorities:
   - Which recommendations should be prioritized in this setting
   - Which can be modified or delayed if resources are limited
   - Alternative approaches when ideal resources aren't available

3. Setting-Specific Guidance:
   - Practical workflow integration tips
   - Training requirements for this setting
   - Quality metrics appropriate for this context
   - Referral considerations (when to refer to higher levels of care)

## Output Format
Format your response as follows:

## Contextual Adaptations

### 1. Resource-Limited Settings

#### Key Adaptations
- [List 3-5 key modifications needed for resource-limited settings]

#### Priority Recommendations
| Recommendation | Adaptation | Resource Requirements |
|----------------|------------|------------------------|
| [Recommendation 1] | [How to adapt] | [Minimal resources needed] |
| [Recommendation 2] | [How to adapt] | [Minimal resources needed] |
| [Continue for key recommendations] |

#### Implementation Guidance
- [Specific implementation tips for this setting]
- [Training considerations]
- [Quality metrics appropriate for this setting]

### 2. Primary Care Settings
[Follow same format as above]

### 3. Specialty Care Settings
[Follow same format as above]
"""

    try:
        # Run the agent
        logger.info(f"Generating context adaptations for {topic} using {'fast' if use_fast_research else 'deep'} research...")
        response = agent.run(prompt)

        # Process response
        if response and response.content:
            result = str(response.content)
            logger.info(f"Generated context adaptations: {len(result)} chars")

            # Cache the result
            if use_cache:
                try:
                    cache = {}
                    if os.path.exists(cache_file):
                        with open(cache_file, "r") as f:
                            cache = json.load(f)
                    cache[cache_key] = result
                    with open(cache_file, "w") as f:
                        json.dump(cache, f)
                except Exception as e:
                    logger.error(f"Error writing to cache: {str(e)}")

            return result
        else:
            logger.error(f"Failed to generate context adaptations for {topic}")
            return "## Contextual Adaptations\n\nNo setting-specific adaptations could be generated. Please check the logs for details."
    except Exception as e:
        logger.error(f"Error generating context adaptations: {str(e)}")
        return f"## Contextual Adaptations\n\nAn error occurred while generating setting-specific adaptations: {str(e)}"

def extract_recommendations(sections_content):
    """Extract key recommendations from sections content for context adaptation"""
    # Look for recommendations in the content
    recommendations = []
    
    # Look for table rows that contain recommendations
    table_rows = re.findall(r'\|.*\|.*\|', sections_content)
    for row in table_rows:
        if 'Updated Recommendation' in row or 'Grade' in row:
            continue  # Skip header rows
        if '**' in row:  # Look for bold text which indicates changes
            recommendations.append(row.strip())
    
    # If no recommendations found in tables, look for numbered lists
    if not recommendations:
        lines = sections_content.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^\d+\.\s+', line.strip()):
                recommendations.append(line.strip())
    
    # If we found more than 10 recommendations, just take the first 10
    if len(recommendations) > 10:
        recommendations = recommendations[:10]
    
    # If we found at least some recommendations, format them nicely
    if recommendations:
        return "Key recommendations from the guidelines:\n\n" + "\n".join(recommendations)
    
    # If all else fails, just provide a summary
    return f"Guidelines for {sections_content.split('###')[0].strip() if '###' in sections_content else 'the medical topic'}"

def extract_numbered_references(content):
    """Extract numbered references from the document"""
    # Look for patterns like "[1] Author et al..." or "1. Author et al..."
    lines = content.split("\n")
    references = []
    ref_pattern1 = re.compile(r'^\d+\.\s+.+')  # Matches "1. Reference..."
    ref_pattern2 = re.compile(r'^\[\d+\]\s+.+')  # Matches "[1] Reference..."
    
    for line in lines:
        if ref_pattern1.match(line.strip()) or ref_pattern2.match(line.strip()):
            if not line.strip().startswith('1. **"') and "**" not in line:  # Avoid capturing recommendation numbers
                references.append(line.strip())
    
    # Look for any "References" section with numeric items
    if "References" in content:
        ref_sections = content.split("References")
        for ref_section in ref_sections[1:]:  # Skip the text before "References"
            section_lines = ref_section.split("\n")
            for line in section_lines:
                if ref_pattern1.match(line.strip()) or ref_pattern2.match(line.strip()):
                    references.append(line.strip())
    
    # Remove duplicates
    unique_references = []
    seen = set()
    for ref in references:
        if ref not in seen:
            unique_references.append(ref)
            seen.add(ref)
    
    return "\n".join(unique_references)

def extract_references_from_content(content):
    """Extract references section from content if it exists"""
    if not content:
        return ""
        
    if "## References" in content:
        return content.split("## References")[1]
    elif "### References" in content:
        return content.split("### References")[1]
    return ""

def remove_references_section(content):
    """Remove references section from content"""
    if not content:
        return ""
        
    if "## References" in content:
        return content.split("## References")[0]
    elif "### References" in content:
        return content.split("### References")[0]
    return content

def combine_references(reference_lists):
    """Combine multiple reference lists into one, removing duplicates"""
    # Join all references
    combined = "\n\n".join([ref for ref in reference_lists if ref])
    
    # Simple deduplication by line
    if combined:
        lines = combined.split("\n")
        unique_lines = []
        seen = set()
        
        for line in lines:
            # Only add non-empty lines that haven't been seen before
            if line.strip() and line not in seen:
                unique_lines.append(line)
                seen.add(line)
        
        return "\n".join(unique_lines)
    
    return ""

# Add a new function to assemble the document with adaptations
def assemble_complete_guidelines_with_adaptations(metadata, sections_content, new_recommendations, conclusion, context_adaptations):
    """Assemble all parts into a complete guidelines document including context adaptations"""
    # Extract title section from metadata (without references)
    title_section = metadata.split("## References")[0] if "## References" in metadata else metadata
    
    # Extract references section from metadata
    references_section = ""
    if "## References" in metadata:
        references_section = metadata.split("## References")[1]
    
    # If there are no references in metadata, look in other sections
    if not references_section or references_section.strip() == "":
        refs_from_sections = extract_references_from_content(sections_content)
        refs_from_new_recs = extract_references_from_content(new_recommendations) if new_recommendations else ""
        refs_from_conclusion = extract_references_from_content(conclusion) if conclusion else ""
        refs_from_context = extract_references_from_content(context_adaptations) if context_adaptations else ""
        
        # Combine all references found
        all_refs = combine_references([refs_from_sections, refs_from_new_recs, refs_from_conclusion, refs_from_context])
        
        if all_refs:
            references_section = all_refs
    
    # Final check - if we still have no references, use a placeholder
    if not references_section or references_section.strip() == "":
        # Try to extract any numbered references from the document
        numbered_refs = extract_numbered_references(title_section + "\n" + sections_content)
        if numbered_refs:
            references_section = numbered_refs
        else:
            references_section = "[References will be added here]"

    # Clean up the sections content but preserve tables
    clean_sections_content = remove_references_section(sections_content).strip()
    clean_new_recommendations = remove_references_section(new_recommendations).strip() if new_recommendations else ""
    clean_conclusion = remove_references_section(conclusion).strip() if conclusion else ""
    clean_context_adaptations = remove_references_section(context_adaptations).strip() if context_adaptations else ""
    
    # Create table of contents
    toc = create_table_of_contents_with_adaptations(
        clean_sections_content, 
        clean_new_recommendations, 
        clean_conclusion,
        clean_context_adaptations
    )
    
    # Assemble the document with enhanced formatting
    document = title_section.strip() + "\n\n"
    document += "---\n\n"  # Add separator
    document += toc + "\n\n"
    document += "---\n\n"  # Add separator
    
    document += "## 📋 Side-by-Side Comparison of Recommendations\n\n"
    document += clean_sections_content + "\n\n"
    document += "---\n\n"  # Add separator

    if clean_new_recommendations:
        document += "## 🆕 New Recommendations\n\n"
        document += clean_new_recommendations + "\n\n"
        document += "---\n\n"  # Add separator

    if clean_conclusion:
        document += "## 📝 Conclusion and Implementation\n\n"
        document += clean_conclusion + "\n\n"
        document += "---\n\n"  # Add separator
        
    # Add context adaptations if available
    if clean_context_adaptations:
        document += "## 🏥 Setting-Specific Adaptations\n\n"
        document += clean_context_adaptations + "\n\n"
        document += "---\n\n"  # Add separator

    # Always add references at the end with proper formatting
    document += "## 📚 References\n\n" + references_section.strip()

    return document

def create_table_of_contents_with_adaptations(sections_content, new_recommendations, conclusion, context_adaptations):
    """Create a table of contents for the document including context adaptations"""
    toc = "## Table of Contents\n\n"
    
    # Add section for recommendations comparison
    toc += "1. [Side-by-Side Comparison of Recommendations](#side-by-side-comparison-of-recommendations)\n"
    
    # Extract section names from sections_content
    section_names = re.findall(r'### ([^\n]+)', sections_content)
    for i, section in enumerate(section_names):
        # Create anchor from section name
        anchor = section.lower().replace(' ', '-').replace('(', '').replace(')', '')
        toc += f"   - [{section}](#{anchor})\n"
    
    # Initialize counter for the remaining sections
    section_num = 2
    
    # Add new recommendations if present
    if new_recommendations:
        toc += f"{section_num}. [New Recommendations](#new-recommendations)\n"
        # Extract new recommendation categories
        new_rec_categories = re.findall(r'### ([^\n]+)', new_recommendations)
        for i, category in enumerate(new_rec_categories):
            # Create anchor from category name
            anchor = category.lower().replace(' ', '-').replace('(', '').replace(')', '')
            toc += f"   - [{category}](#{anchor})\n"
        section_num += 1
    
    # Add conclusion if present
    if conclusion:
        toc += f"{section_num}. [Conclusion and Implementation](#conclusion-and-implementation)\n"
        
        # Extract conclusion subsections
        conclusion_subsections = re.findall(r'### ([^\n]+)', conclusion)
        for i, subsection in enumerate(conclusion_subsections):
            # Create anchor from subsection name
            anchor = subsection.lower().replace(' ', '-').replace('(', '').replace(')', '')
            toc += f"   - [{subsection}](#{anchor})\n"
        section_num += 1
    
    # Add context adaptations if present
    if context_adaptations:
        toc += f"{section_num}. [Setting-Specific Adaptations](#setting-specific-adaptations)\n"
        
        # Extract adaptation subsections
        adaptation_subsections = re.findall(r'### (\d+\.\s+[^\n]+)', context_adaptations)
        for i, subsection in enumerate(adaptation_subsections):
            # Create anchor from subsection name
            clean_subsection = subsection.split('.', 1)[1].strip()
            anchor = clean_subsection.lower().replace(' ', '-').replace('(', '').replace(')', '')
            toc += f"   - [{clean_subsection}](#{anchor})\n"
        section_num += 1
    
    # Add references
    toc += f"{section_num}. [References](#references)\n"
    
    return toc

# Streamlit App
def main():
    st.set_page_config(
        page_title="Advanced Medical Guideline Updater",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🩺 Advanced Medical Guideline Updater")
    st.markdown(
        """
        ### AI-Powered Evidence-Based Guidelines Platform

        This advanced platform automatically researches, analyzes, and synthesizes medical guidelines with the latest clinical evidence to:

        1. **Find authoritative guidelines** - Identifies the most recent major society guidelines
        2. **Discover practice-changing evidence** - Locates high-impact studies published since guideline release
        3. **Provide evidence-based updates** - Creates detailed, actionable recommendation updates
        4. **Develop new recommendations** - Addresses clinical areas not covered in original guidelines
        5. **Support implementation** - Offers practical guidance for clinical adoption
        6. **Adapt to different settings** - Provides context-specific versions for different healthcare environments

        Powered by Perplexity's AI research models.
        """
    )

    # Create tmp directory if it doesn't exist
    os.makedirs("tmp", exist_ok=True)

    # Sidebar
    with st.sidebar:
        st.header("About This Platform")
        st.markdown("""
        This platform demonstrates how AI deep research can accelerate the medical guideline update process by:

        - Continuously monitoring emerging medical literature
        - Identifying potentially practice-changing evidence
        - Performing rigorous evidence evaluation
        - Creating structured, transparent updates
        - Providing implementation guidance
        - Customizing recommendations for different healthcare settings

        *Note: This is a demonstration tool. All outputs should be reviewed by medical experts before implementation.*
        """)

        st.header("Options")
        topic = st.text_input("Medical Topic", "fever evaluation in critically ill patients")

        # Research Mode Selection
        st.subheader("Research Mode")
        research_mode = st.radio(
            "Select research depth:",
            ["Deep Research (Comprehensive, Slower)", "Fast Research (Quicker, Less Detailed)"],
            help="Deep Research uses sonar-deep-research for thorough analysis. Fast Research uses sonar-pro for quicker results."
        )
        # Convert to boolean for easier use in functions
        use_fast_research = research_mode.startswith("Fast")

        # Define the sections to research
        with st.expander("Advanced Configuration"):
            sections = st.multiselect(
                "Clinical Sections to Include",
                ["Temperature Measurement", "Diagnostic Approach", "Microbiological Evaluation",
                 "Imaging Studies", "Biomarker Testing", "Non-Infectious Causes",
                 "Special Populations", "Treatment Considerations"],
                default=["Temperature Measurement", "Diagnostic Approach", "Microbiological Evaluation", "Imaging Studies"]
            )

            include_new = st.checkbox("Include New Recommendations", value=True)
            include_conclusion = st.checkbox("Generate Comprehensive Conclusion", value=True)
            # NEW: Add contextual adaptation option
            include_context_adaptations = st.checkbox("Generate Setting-Specific Adaptations", value=True,
                                                      help="Creates context-specific versions for different healthcare settings")
            use_cache = st.checkbox("Use Cached Results (if available)", value=True)
            chunked_generation = st.checkbox("Use Chunked Generation for Long Outputs", value=True,
                                             help="Breaks generation into smaller pieces to avoid truncation")

        submit = st.button("Generate Updated Guidelines", type="primary")

    # Main content area with three columns for better organization
    col1, col2, col3 = st.columns([1, 2, 2])

    with col1:
        st.subheader("Research Process")
        process_container = st.container()

        # Display API key status
        if perplexity_api_key:
            st.success("✅ Perplexity API key detected")
        else:
            st.error("❌ Perplexity API key not found")
            st.info("Set your API key with: setx PERPLEXITY_API_KEY your_key")

        # Display selected research mode
        st.info(f"Using {'Fast Research (sonar-pro)' if use_fast_research else 'Deep Research (sonar-deep-research)'}")

    with col2:
        st.subheader("Generation Progress")
        progress_container = st.container()

    with col3:
        st.subheader("Updated Guidelines")
        result_container = st.empty()

    if submit:
        # Clear the result container
        result_container.empty()

        with process_container:
            st.write("**Research & Generation Process:**")
            status_log = st.empty()

        with progress_container:
            # Initialize progress bar with float 0.0
            progress = st.progress(0.0)
            status = st.empty()

            # Create expandable areas for detailed progress
            metadata_expander = st.expander("1. Executive Summary", expanded=False)
            sections_expander = st.expander("2. Clinical Recommendations", expanded=False)
            new_recs_expander = st.expander("3. New Recommendations", expanded=False) if include_new else None
            conclusion_expander = st.expander("4. Conclusion & Implementation", expanded=False) if include_conclusion else None
            # NEW: Add contextual adaptations expander
            context_expander = st.expander("5. Setting-Specific Adaptations", expanded=False) if include_context_adaptations else None

        # Initialize a tracker for completed parts
        completed_parts = []

        try:
            # Status update function with the fix for st.progress
            def update_status(message, progress_value):
                status.markdown(f"### {message}")
                # FIX: Divide progress_value by 100.0 to get value between 0.0 and 1.0
                progress.progress(progress_value / 100.0)
                status_log.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}")
                time.sleep(0.1) # Reduced sleep time for potentially faster updates

            # Helper function to handle section research with progress tracking
            def research_section_with_progress(topic, section, section_index, total_sections, use_fast_research=False, use_cache=True):
                # Calculate progress percentage for this section
                # Adjust progress allocation to accommodate contextual adaptations
                section_progress_total = 50 if include_context_adaptations else 60
                start_percent = 20 + (section_index * section_progress_total / total_sections)
                end_percent = 20 + ((section_index + 1) * section_progress_total / total_sections)

                # Update status
                update_status(f"Researching {section} section...", start_percent)

                # Execute research
                if chunked_generation:
                    # Break the research into smaller chunks for complex sections
                    result = research_section_chunked(topic, section, use_fast_research, use_cache)
                else:
                    # Use the standard function
                    result = research_guideline_section(topic, section, use_fast_research, use_cache)

                # Update progress
                # Ensure the progress reaches the end of the step even if the function returns quickly
                update_status(f"Completed {section} section", end_percent)

                return result

            # Added a check here as the agent creation might fail without the API key
            if not perplexity_api_key and 'Agent' in globals() and 'Perplexity' in globals():
                st.error("PERPLEXITY_API_KEY is not set. Please set the environment variable to proceed.")
                update_status("Generation Failed", 0) # Reset progress on error
                return # Stop execution if API key is missing

            # STEP 1: Research metadata and executive summary
            # Allocate 20% progress to metadata/exec summary
            update_status("Researching guideline metadata and writing executive summary...", 10)

            # Generate metadata (executive summary, etc.)
            metadata_result = research_guideline_metadata(topic, use_fast_research, use_cache)
            completed_parts.append("metadata")

            # Display in expander
            with metadata_expander:
                st.markdown(metadata_result)

            # Ensure progress reaches 20% after this step
            update_status("Executive summary complete", 20)

            # STEP 2: Research each section separately
            sections_results = []
            total_sections = len(sections)
            # Check if there are sections to process to avoid potential division by zero
            if total_sections == 0:
                st.warning("No clinical sections selected to research.")
            else:
                for i, section in enumerate(sections):
                    section_result = research_section_with_progress(
                        topic, section, i, total_sections, use_fast_research, use_cache
                    )
                    sections_results.append(section_result)

                    # Display in expander
                    with sections_expander:
                        st.markdown(section_result)

                    # Small delay to avoid rate limiting
                    time.sleep(0.5)

            # Combine all section results
            combined_sections = "\n\n".join(sections_results)
            completed_parts.append("sections")

            # Adjust progress allocations based on included components
            progress_allocation = {}
            remaining_progress = 100 - (20 + (50 if include_context_adaptations else 60))
            components_count = sum([include_new, include_conclusion, include_context_adaptations])
            
            if components_count > 0:
                progress_per_component = remaining_progress / components_count
                if include_new:
                    progress_allocation["new_recommendations"] = progress_per_component
                if include_conclusion:
                    progress_allocation["conclusion"] = progress_per_component
                if include_context_adaptations:
                    progress_allocation["context_adaptations"] = progress_per_component

            # Calculate starting progress for each component
            current_progress = 20 + (50 if include_context_adaptations else 60)
            
            # STEP 3: Research new recommendations if requested
            new_recommendations_result = ""
            if include_new:
                # Use allocated progress for new recommendations
                start_progress = current_progress
                end_progress = current_progress + progress_allocation["new_recommendations"]
                current_progress = end_progress
                
                update_status("Researching potential new recommendations...", start_progress)

                # Generate new recommendations
                if chunked_generation:
                    new_recommendations_result = research_new_recommendations_chunked(topic, use_fast_research, use_cache)
                else:
                    new_recommendations_result = research_new_recommendations(topic, use_fast_research, use_cache)

                completed_parts.append("new_recommendations")

                # Display in expander (check if new_recs_expander exists)
                if new_recs_expander:
                    with new_recs_expander:
                        st.markdown(new_recommendations_result)

                # Update progress
                update_status("New recommendations complete", end_progress)

            # STEP 4: Generate conclusion if requested
            conclusion_result = ""
            if include_conclusion:
                # Use allocated progress for conclusion
                start_progress = current_progress
                end_progress = current_progress + progress_allocation["conclusion"]
                current_progress = end_progress
                
                update_status("Creating comprehensive conclusion...", start_progress)

                # Generate conclusion
                conclusion_result = research_comprehensive_conclusion(
                    topic, metadata_result, combined_sections, use_fast_research, use_cache
                )

                completed_parts.append("conclusion")

                # Display in expander (check if conclusion_expander exists)
                if conclusion_expander:
                    with conclusion_expander:
                        st.markdown(conclusion_result)

                # Update progress
                update_status("Conclusion complete", end_progress)

            # STEP 5: Generate contextual adaptations if requested
            context_adaptations_result = ""
            if include_context_adaptations:
                # Use allocated progress for contextual adaptations
                start_progress = current_progress
                end_progress = current_progress + progress_allocation["context_adaptations"]
                current_progress = end_progress
                
                update_status("Generating setting-specific adaptations...", start_progress)

                # Generate contextual adaptations
                context_adaptations_result = generate_context_adaptations(
                    topic, combined_sections, use_fast_research, use_cache
                )

                completed_parts.append("context_adaptations")

                # Display in expander (check if context_expander exists)
                if context_expander:
                    with context_expander:
                        st.markdown(context_adaptations_result)

                # Update progress
                update_status("Setting-specific adaptations complete", end_progress)

            # FINAL STEP: Assemble the complete document
            # Allocate remaining progress to assembly and final display
            update_status("Assembling complete guidelines document...", 98)

            # Assemble document based on completed parts
            complete_document = assemble_complete_guidelines_with_adaptations(
                metadata_result,
                combined_sections,
                new_recommendations_result,
                conclusion_result,
                context_adaptations_result
            )

            # Final update
            update_status("✅ Guidelines research and update complete!", 100)

            # Show document information
            st.write(f"Document length: {len(complete_document)} characters")
            st.write(f"Completed parts: {', '.join(completed_parts)}")

            # Store in session state for download
            st.session_state["markdown_content"] = complete_document

            # Display final result
            result_container.markdown(complete_document)

            # Download buttons
            st.subheader("Download Options")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="📥 Download as Markdown",
                    data=complete_document,
                    file_name=f"{topic.replace(' ', '_')}_guideline_update.md",
                    mime="text/markdown",
                )

            with col_dl2:
                # Convert to HTML for better printing
                try:
                    import markdown
                    html_content = markdown.markdown(complete_document)
                    st.download_button(
                        label="📄 Download as HTML",
                        data=html_content,
                        file_name=f"{topic.replace(' ', '_')}_guideline_update.html",
                        mime="text/html",
                    )
                except ImportError:
                    st.warning("Install 'markdown' library (`pip install markdown`) for HTML download option.")
                except Exception as e:
                    st.error(f"Could not create HTML version: {str(e)}")

            # Save file with proper encoding
            try:
                # Use a safer filename by replacing non-alphanumeric chars
                safe_topic = re.sub(r'[^\w.-]', '_', topic)
                output_filename = f"tmp/{safe_topic}_guideline_update.md"
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(complete_document)
                st.success(f"Output saved to {output_filename}")
            except Exception as e:
                st.error(f"Could not save output file: {str(e)}")
                # Fallback to ASCII saving with error ignoring
                try:
                    output_filename_ascii = f"tmp/{safe_topic}_guideline_update_ascii.md"
                    with open(output_filename_ascii, "w", encoding="ascii", errors="ignore") as f:
                        f.write(complete_document)
                    st.warning(f"Output saved with ASCII encoding (some characters may be lost) to {output_filename_ascii}")
                except Exception as e2:
                    st.error(f"Could not save output with ASCII encoding: {str(e2)}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Workflow error: {str(e)}")
            update_status("Generation Failed", 0) # Ensure progress bar resets on error

if __name__ == "__main__":
    main()
