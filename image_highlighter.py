"""
Image Highlighting Module
Adds visual highlighting to concepts in the original image
"""
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple
import io


class ImageHighlighter:
    """Handles highlighting concepts in images"""
    
    # Color palette for different concepts (RGB)
    COLORS = [
        (255, 200, 0),      # Yellow
        (0, 200, 255),      # Cyan
        (255, 100, 150),    # Pink
        (100, 255, 100),    # Light Green
        (255, 150, 0),      # Orange
        (200, 100, 255),    # Purple
        (100, 200, 255),    # Light Blue
        (255, 200, 200),    # Light Red
    ]
    
    def __init__(self, thickness: int = 3, alpha: float = 0.3):
        """
        Initialize the highlighter
        
        Args:
            thickness: Border thickness in pixels
            alpha: Transparency level (0.0-1.0)
        """
        self.thickness = thickness
        self.alpha = alpha
    
    def _get_color_for_index(self, index: int) -> Tuple[int, int, int]:
        """Get a color for the given concept index"""
        return self.COLORS[index % len(self.COLORS)]
    
    def _denormalize_region(self, region: Dict, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Convert normalized region (0-1) to pixel coordinates
        
        Args:
            region: Dict with x1, y1, x2, y2 (normalized to 0-1)
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates
        """
        x1 = int(region.get('x1', 0) * width)
        y1 = int(region.get('y1', 0) * height)
        x2 = int(region.get('x2', 1) * width)
        y2 = int(region.get('y2', 1) * height)
        
        # Ensure valid coordinates
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        return x1, y1, x2, y2
    
    def highlight_concepts(
        self,
        image_base64: str,
        concepts: List[Dict]
    ) -> Tuple[str, List[Dict]]:
        """
        Highlight concepts in the image
        
        Args:
            image_base64: Base64 encoded image
            concepts: List of concept dictionaries with 'region' data
            
        Returns:
            Tuple of (highlighted_image_base64, concept_regions_with_colors)
        """
        try:
            # Decode image
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            width, height = image.size
            
            # Create a copy for highlighting
            highlighted_image = image.copy()
            draw = ImageDraw.Draw(highlighted_image)
            
            # Store region info with colors
            concept_regions = []
            
            # Draw rectangles for each concept
            for idx, concept in enumerate(concepts):
                if 'region' not in concept:
                    continue
                
                region = concept['region']
                color = self._get_color_for_index(idx)
                
                # Denormalize coordinates
                x1, y1, x2, y2 = self._denormalize_region(region, width, height)
                
                # Draw rectangle
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline=color,
                    width=self.thickness
                )
                
                # Store region info for frontend
                concept_regions.append({
                    "id": concept.get('id', f"concept_{idx}"),
                    "name": concept.get('name', 'Unknown'),
                    "summary": concept.get('summary', ''),
                    "color": {
                        "r": color[0],
                        "g": color[1],
                        "b": color[2]
                    },
                    "region": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "normalized_region": region,
                    "category": concept.get('category', 'General')
                })
            
            # Encode highlighted image back to base64
            buffer = BytesIO()
            highlighted_image.save(buffer, format='PNG')
            highlighted_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return highlighted_base64, concept_regions
            
        except Exception as e:
            raise ValueError(f"Failed to highlight image: {str(e)}")
    
    def create_svg_overlay(
        self,
        image_width: int,
        image_height: int,
        concept_regions: List[Dict]
    ) -> str:
        """
        Create an SVG overlay for interactive regions
        
        Args:
            image_width: Width of the image
            image_height: Height of the image
            concept_regions: List of concept regions with coordinates
            
        Returns:
            SVG string as an overlay
        """
        svg_parts = [
            f'<svg width="{image_width}" height="{image_height}" style="position: absolute; top: 0; left: 0;">',
        ]
        
        for region in concept_regions:
            rect = region['region']
            color = region['color']
            rgb = f"rgb({color['r']}, {color['g']}, {color['b']})"
            concept_id = region['id']
            
            svg_parts.append(
                f'<rect x="{rect["x1"]}" y="{rect["y1"]}" '
                f'width="{rect["x2"] - rect["x1"]}" height="{rect["y2"] - rect["y1"]}" '
                f'class="concept-region" data-concept-id="{concept_id}" '
                f'style="fill: {rgb}; opacity: 0.1; cursor: pointer; stroke: {rgb}; stroke-width: 2;" '
                f'onclick="handleConceptClick(\'{concept_id}\')" />'
            )
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)


def highlight_image_with_concepts(
    image_base64: str,
    concepts: List[Dict]
) -> Dict:
    """
    Convenience function to highlight an image with concepts
    
    Args:
        image_base64: Base64 encoded image
        concepts: List of concept dictionaries
        
    Returns:
        Dictionary with highlighted image and region data
    """
    highlighter = ImageHighlighter()
    
    try:
        highlighted_base64, concept_regions = highlighter.highlight_concepts(
            image_base64,
            concepts
        )
        
        return {
            "success": True,
            "highlighted_image": highlighted_base64,
            "regions": concept_regions,
            "image_dimensions": {
                # These will be populated from the original image
                "width": Image.open(BytesIO(base64.b64decode(image_base64))).width,
                "height": Image.open(BytesIO(base64.b64decode(image_base64))).height
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "highlighted_image": None,
            "regions": []
        }
