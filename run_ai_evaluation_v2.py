#!/usr/bin/env python3
"""
4-AI Evaluation Script v2
==========================
정확한 모델명을 사용하여 4개 AI 모델에게 평가 요청

평가 모델:
1. OpenAI GPT-4 (gpt-4o)
2. Anthropic Claude (claude-opus-4-1-20250805)
3. Google Gemini (gemini-2.5-pro)
4. xAI Grok (grok-4-latest) - REST API
"""

import os
import json
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Evaluation prompt
EVALUATION_PROMPT = Path("AI_EVALUATION_PROMPT.md").read_text()

# Context files to include
CONTEXT_FILES = [
    "PHASE1_VALIDATION_REPORT.md",
    "README.md",
    "pyproject.toml",
    "config/ares7_qm_turbo_final_251129.yaml",
]


def load_context() -> str:
    """Load context files"""
    context = []
    
    for file_path in CONTEXT_FILES:
        path = Path(file_path)
        if path.exists():
            context.append(f"\n\n## File: {file_path}\n\n```\n{path.read_text()}\n```")
    
    return "\n".join(context)


async def evaluate_with_openai():
    """Evaluate with OpenAI GPT-4"""
    print("\n" + "=" * 80)
    print("Evaluating with OpenAI GPT-4 (gpt-4o)")
    print("=" * 80)
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        context = load_context()
        full_prompt = f"{EVALUATION_PROMPT}\n\n{context}"
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert code reviewer and deployment specialist for quantitative trading systems."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content
        
        # Save result
        output_path = Path("ai_evaluations/openai_gpt4_evaluation.md")
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(result)
        
        print(f"✅ OpenAI evaluation saved to: {output_path}")
        print(result[:500] + "...")
        
        return result
        
    except Exception as e:
        print(f"❌ OpenAI evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def evaluate_with_anthropic():
    """Evaluate with Anthropic Claude"""
    print("\n" + "=" * 80)
    print("Evaluating with Anthropic Claude (claude-opus-4-1-20250805)")
    print("=" * 80)
    
    try:
        from anthropic import Anthropic
        
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        context = load_context()
        full_prompt = f"{EVALUATION_PROMPT}\n\n{context}"
        
        # Use the exact model name provided
        response = client.messages.create(
            model="claude-opus-4-1-20250805",  # Exact model name provided
            max_tokens=4000,
            temperature=0.3,
            system="You are an expert code reviewer and deployment specialist for quantitative trading systems.",
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        
        result = response.content[0].text
        
        # Save result
        output_path = Path("ai_evaluations/anthropic_claude_evaluation.md")
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(result)
        
        print(f"✅ Anthropic evaluation saved to: {output_path}")
        print(result[:500] + "...")
        
        return result
        
    except Exception as e:
        print(f"❌ Anthropic evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def evaluate_with_gemini():
    """Evaluate with Google Gemini using REST API"""
    print("\n" + "=" * 80)
    print("Evaluating with Google Gemini (gemini-2.5-pro)")
    print("=" * 80)
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={api_key}"
        
        context = load_context()
        full_prompt = f"{EVALUATION_PROMPT}\n\n{context}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 4000,
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data['candidates'][0]['content']['parts'][0]['text']
                    
                    # Save result
                    output_path = Path("ai_evaluations/google_gemini_evaluation.md")
                    output_path.parent.mkdir(exist_ok=True)
                    output_path.write_text(result)
                    
                    print(f"✅ Gemini evaluation saved to: {output_path}")
                    print(result[:500] + "...")
                    
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ Gemini API error {response.status}: {error_text}")
                    return None
        
    except Exception as e:
        print(f"❌ Gemini evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def evaluate_with_grok():
    """Evaluate with xAI Grok using REST API"""
    print("\n" + "=" * 80)
    print("Evaluating with xAI Grok (grok-4-latest)")
    print("=" * 80)
    
    try:
        api_key = os.getenv("XAI_API_KEY")
        url = "https://api.x.ai/v1/chat/completions"
        
        context = load_context()
        full_prompt = f"{EVALUATION_PROMPT}\n\n{context}"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert code reviewer and deployment specialist for quantitative trading systems."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "model": "grok-4-latest",  # Exact model name provided
            "stream": False,
            "temperature": 0.3
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    result = data['choices'][0]['message']['content']
                    
                    # Save result
                    output_path = Path("ai_evaluations/xai_grok_evaluation.md")
                    output_path.parent.mkdir(exist_ok=True)
                    output_path.write_text(result)
                    
                    print(f"✅ Grok evaluation saved to: {output_path}")
                    print(result[:500] + "...")
                    
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ Grok API error {response.status}: {error_text}")
                    return None
        
    except Exception as e:
        print(f"❌ Grok evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_score(evaluation_text: str) -> int:
    """Parse total score from evaluation text"""
    import re
    
    # Look for "총점: XX/100" or "Total Score: XX/100"
    patterns = [
        r'총점[:\s]+(\d+)/100',
        r'Total Score[:\s]+(\d+)/100',
        r'TOTAL[:\s]+(\d+)/100',
        r'##\s+총점[:\s]+(\d+)/100',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, evaluation_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return 0


async def main():
    """Run all evaluations"""
    print("=" * 80)
    print("ARES-Ultimate-251129 4-AI Evaluation v2")
    print("=" * 80)
    print()
    print("Evaluating deployment readiness with 4 AI models...")
    print("Using exact model names:")
    print("  - gpt-4o")
    print("  - claude-opus-4-20250514")
    print("  - gemini-2.0-flash-exp")
    print("  - grok-2-latest")
    print()
    
    # Run evaluations sequentially to avoid rate limits
    results = {}
    
    results['openai'] = await evaluate_with_openai()
    await asyncio.sleep(2)
    
    results['anthropic'] = await evaluate_with_anthropic()
    await asyncio.sleep(2)
    
    results['gemini'] = await evaluate_with_gemini()
    await asyncio.sleep(2)
    
    results['grok'] = await evaluate_with_grok()
    
    # Calculate scores
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    
    scores = {}
    for model, result in results.items():
        if result:
            score = parse_score(result)
            scores[model] = score
            print(f"{model.upper():15s}: {score}/100")
        else:
            print(f"{model.upper():15s}: FAILED")
    
    if scores:
        avg_score = sum(scores.values()) / len(scores)
        print("-" * 80)
        print(f"{'AVERAGE':15s}: {avg_score:.1f}/100")
        print()
        
        if avg_score >= 95:
            print("✅ PASSED: Average score >= 95, ready for EC2 deployment!")
        elif avg_score >= 90:
            print("⚠️  CONDITIONAL: Average score 90-94, improvements recommended")
        else:
            print("❌ FAILED: Average score < 90, major improvements required")
    else:
        print("\n❌ All evaluations failed")
    
    print("\n" + "=" * 80)
    print("Evaluation reports saved in: ai_evaluations/")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
