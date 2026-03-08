"""
Evaluation Module for VisualVerse
Evaluates the quality of comic and mindmap outputs
"""

import os
import sys
import json
from typing import Dict, List, Any
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))


class ComicEvaluator:
    """
    Evaluate comic generation quality
    
    Metrics:
    - Story alignment score: How well panels follow the story
    - Scene relevance: Relevance of panel descriptions
    - Panel consistency: Character/setting consistency across panels
    - Visual coherence: Overall visual narrative flow
    """
    
    def evaluate(self, original_text: str, comic_result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate comic generation quality"""
        panels = comic_result.get("panels", [])
        
        if not panels:
            return {"error": "No panels to evaluate"}
        
        scores = {
            "story_alignment": self._evaluate_story_alignment(original_text, panels),
            "scene_relevance": self._evaluate_scene_relevance(original_text, panels),
            "panel_consistency": self._evaluate_panel_consistency(panels),
            "visual_coherence": self._evaluate_visual_coherence(panels)
        }
        
        # Calculate overall score
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _evaluate_story_alignment(self, text: str, panels: List[Dict]) -> float:
        """Check if panels follow the story sequence"""
        text_lower = text.lower()
        alignment_scores = []
        
        for panel in panels:
            caption = panel.get("caption", "").lower()
            # Check word overlap
            text_words = set(text_lower.split())
            caption_words = set(caption.split())
            overlap = len(text_words & caption_words)
            score = min(1.0, overlap / max(len(caption_words), 1))
            alignment_scores.append(score)
        
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0
    
    def _evaluate_scene_relevance(self, text: str, panels: List[Dict]) -> float:
        """Evaluate if scene descriptions are relevant"""
        relevance_scores = []
        
        for panel in panels:
            prompt = panel.get("prompt", "")
            caption = panel.get("caption", "")
            
            # Check if prompt relates to caption
            prompt_words = set(prompt.lower().split())
            caption_words = set(caption.lower().split())
            overlap = len(prompt_words & caption_words)
            score = min(1.0, overlap / max(len(caption_words), 1) * 2)
            relevance_scores.append(score)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    def _evaluate_panel_consistency(self, panels: List[Dict]) -> float:
        """Check character/setting consistency across panels"""
        if len(panels) < 2:
            return 1.0
        
        # Collect all characters and settings
        all_characters = []
        all_settings = []
        
        for panel in panels:
            all_characters.extend(panel.get("characters", []))
            setting = panel.get("setting", "")
            if setting:
                all_settings.append(setting)
        
        # Check consistency
        if not all_characters and not all_settings:
            return 0.5  # Neutral if no characters/settings detected
        
        # Characters should repeat across panels
        char_consistency = 1.0 if len(set(all_characters)) <= len(panels) else 0.7
        
        return char_consistency
    
    def _evaluate_visual_coherence(self, panels: List[Dict]) -> float:
        """Evaluate visual narrative flow"""
        if len(panels) < 2:
            return 1.0
        
        # Check if panels have progressive structure
        has_opening = panels[0].get("panel_number", 0) == 1
        has_sequence = all(
            panels[i].get("panel_number", 0) < panels[i+1].get("panel_number", 0)
            for i in range(len(panels) - 1)
        )
        
        score = 0.0
        if has_opening:
            score += 0.5
        if has_sequence:
            score += 0.5
        
        return score


class MindMapEvaluator:
    """
    Evaluate mind map generation quality
    
    Metrics:
    - Keyphrase accuracy: Relevance of extracted keyphrases
    - Concept clustering: Quality of topic grouping
    - Graph connectivity: How well concepts are connected
    - Hierarchy quality: Logical hierarchy structure
    """
    
    def evaluate(self, original_text: str, mindmap_result: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate mind map generation quality"""
        graph = mindmap_result.get("graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        
        if not nodes:
            return {"error": "No nodes to evaluate"}
        
        scores = {
            "keyphrase_accuracy": self._evaluate_keyphrase_accuracy(original_text, nodes),
            "concept_clustering": self._evaluate_clustering(nodes),
            "graph_connectivity": self._evaluate_connectivity(nodes, edges),
            "hierarchy_quality": self._evaluate_hierarchy(nodes, edges)
        }
        
        # Calculate overall score
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores
    
    def _evaluate_keyphrase_accuracy(self, text: str, nodes: List[Dict]) -> float:
        """Check if extracted keyphrases appear in original text"""
        text_lower = text.lower()
        
        matches = 0
        for node in nodes:
            label = node.get("label", "").lower()
            if label in text_lower or any(word in text_lower for word in label.split()):
                matches += 1
        
        return matches / len(nodes) if nodes else 0.0
    
    def _evaluate_clustering(self, nodes: List[Dict]) -> float:
        """Evaluate quality of topic clustering"""
        # Check if nodes have topic assignments
        topic_counts = {}
        for node in nodes:
            topic_id = node.get("topic_id", "")
            if topic_id:
                topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        
        if not topic_counts:
            return 0.5  # Neutral if no topics
        
        # Good clustering has balanced topic sizes
        sizes = list(topic_counts.values())
        avg_size = sum(sizes) / len(sizes)
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        
        # Lower variance = better clustering
        score = 1.0 / (1.0 + variance / avg_size) if avg_size > 0 else 0.5
        
        return score
    
    def _evaluate_connectivity(self, nodes: List[Dict], edges: List[Dict]) -> float:
        """Evaluate graph connectivity"""
        if len(nodes) < 2:
            return 1.0
        
        # Calculate edge density
        max_edges = len(nodes) * (len(nodes) - 1) / 2
        actual_edges = len(edges)
        
        # Too sparse or too dense is bad
        density = actual_edges / max_edges if max_edges > 0 else 0
        
        # Optimal density is around 0.1-0.3 for readability
        if density < 0.05:
            return 0.3  # Too sparse
        elif density > 0.5:
            return 0.6  # Too dense
        else:
            return 0.8 + 0.2 * (1 - abs(density - 0.2) / 0.2)
    
    def _evaluate_hierarchy(self, nodes: List[Dict], edges: List[Dict]) -> float:
        """Evaluate hierarchy quality"""
        # Check for topic nodes (higher level)
        topic_nodes = [n for n in nodes if n.get("type") == "topic"]
        concept_nodes = [n for n in nodes if n.get("type") != "topic"]
        
        if not topic_nodes:
            return 0.5  # No clear hierarchy
        
        # Check if topics connect to concepts
        topic_ids = {n.get("id") for n in topic_nodes}
        connected_from_topics = sum(
            1 for e in edges if e.get("source") in topic_ids
        )
        
        if concept_nodes:
            hierarchy_score = connected_from_topics / len(concept_nodes)
        else:
            hierarchy_score = 0.5
        
        return min(1.0, hierarchy_score)


class FullPipelineEvaluator:
    """Evaluate the complete VisualVerse pipeline"""
    
    def __init__(self):
        self.comic_evaluator = ComicEvaluator()
        self.mindmap_evaluator = MindMapEvaluator()
    
    def evaluate_pipeline(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate pipeline on test cases
        
        Each test case: {text, expected_mode, ...}
        """
        results = {
            "comic_scores": [],
            "mindmap_scores": [],
            "classification_accuracy": 0,
            "total_cases": len(test_cases)
        }
        
        correct_classifications = 0
        
        for case in test_cases:
            mode = case.get("mode")
            text = case.get("text", "")
            expected_mode = case.get("expected_mode")
            
            if mode == expected_mode:
                correct_classifications += 1
            
            if mode == "comic" and "comic_data" in case:
                scores = self.comic_evaluator.evaluate(text, {"panels": case["comic_data"]})
                results["comic_scores"].append(scores)
            
            if mode == "mindmap" and "mindmap_data" in case:
                scores = self.mindmap_evaluator.evaluate(text, {"graph": case["mindmap_data"]})
                results["mindmap_scores"].append(scores)
        
        results["classification_accuracy"] = correct_classifications / len(test_cases) if test_cases else 0
        
        # Calculate averages
        if results["comic_scores"]:
            results["avg_comic_score"] = sum(s.get("overall", 0) for s in results["comic_scores"]) / len(results["comic_scores"])
        
        if results["mindmap_scores"]:
            results["avg_mindmap_score"] = sum(s.get("overall", 0) for s in results["mindmap_scores"]) / len(results["mindmap_scores"])
        
        return results
    
    def generate_report(self, results: Dict, output_path: str = "evaluation/evaluation_report.txt"):
        """Generate evaluation report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = [
            "=" * 60,
            "VISUALVERSE PIPELINE EVALUATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "CLASSIFICATION",
            "-" * 40,
            f"Accuracy: {results.get('classification_accuracy', 0):.2%}",
            "",
            "COMIC GENERATION",
            "-" * 40,
            f"Average Score: {results.get('avg_comic_score', 0):.2%}",
            f"Cases Evaluated: {len(results.get('comic_scores', []))}",
            "",
            "MINDMAP GENERATION",
            "-" * 40,
            f"Average Score: {results.get('avg_mindmap_score', 0):.2%}",
            f"Cases Evaluated: {len(results.get('mindmap_scores', []))}",
            "",
            "=" * 60
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Report saved to {output_path}")
        return '\n'.join(report)


def main():
    """Run evaluation on sample data"""
    print("=" * 60)
    print(" VisualVerse Evaluation")
    print("=" * 60)
    
    evaluator = FullPipelineEvaluator()
    
    # Sample test cases
    test_cases = [
        {
            "text": "Once upon a time, a young hero set out on an adventure to save the kingdom.",
            "mode": "comic",
            "expected_mode": "comic",
            "comic_data": [
                {"panel_number": 1, "caption": "A young hero prepares for adventure", "prompt": "Hero standing at village gate", "characters": ["hero"], "setting": "village"},
                {"panel_number": 2, "caption": "The journey begins", "prompt": "Hero walking through forest", "characters": ["hero"], "setting": "forest"}
            ]
        },
        {
            "text": "Machine learning is a subset of AI that enables computers to learn from data.",
            "mode": "mindmap",
            "expected_mode": "mindmap",
            "mindmap_data": {
                "nodes": [
                    {"id": "1", "label": "Machine Learning", "type": "topic", "topic_id": "t1"},
                    {"id": "2", "label": "AI", "type": "concept", "topic_id": "t1"},
                    {"id": "3", "label": "Data", "type": "concept", "topic_id": "t1"}
                ],
                "edges": [
                    {"source": "1", "target": "2", "relation": "IS_A"},
                    {"source": "1", "target": "3", "relation": "REQUIRES"}
                ]
            }
        }
    ]
    
    results = evaluator.evaluate_pipeline(test_cases)
    report = evaluator.generate_report(results)
    print(report)


if __name__ == "__main__":
    main()
