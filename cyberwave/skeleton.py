"""
Skeleton data structures for human pose tracking and teleoperation.

These classes provide standardized representations for human body tracking
data used in teleoperation and human-robot interaction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import time


@dataclass
class Joint3D:
    """Represents a 3D joint position with confidence."""
    x: float
    y: float
    z: float
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            'x': self.x,
            'y': self.y, 
            'z': self.z,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> Joint3D:
        """Create Joint3D from dictionary."""
        return cls(
            x=data['x'],
            y=data['y'],
            z=data['z'],
            confidence=data.get('confidence', 1.0)
        )


@dataclass
class HandSkeleton:
    """Represents a complete hand skeleton with 21 landmarks."""
    
    # Hand landmarks (MediaPipe format)
    wrist: Joint3D
    thumb_cmc: Joint3D
    thumb_mcp: Joint3D
    thumb_ip: Joint3D
    thumb_tip: Joint3D
    
    index_mcp: Joint3D
    index_pip: Joint3D
    index_dip: Joint3D
    index_tip: Joint3D
    
    middle_mcp: Joint3D
    middle_pip: Joint3D
    middle_dip: Joint3D
    middle_tip: Joint3D
    
    ring_mcp: Joint3D
    ring_pip: Joint3D
    ring_dip: Joint3D
    ring_tip: Joint3D
    
    pinky_mcp: Joint3D
    pinky_pip: Joint3D
    pinky_dip: Joint3D
    pinky_tip: Joint3D
    
    # Metadata
    hand_type: str = "right"  # "left" or "right"
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_mediapipe_landmarks(cls, landmarks, hand_type: str = "right", timestamp: Optional[float] = None) -> HandSkeleton:
        """Create HandSkeleton from MediaPipe hand landmarks."""
        if timestamp is None:
            timestamp = time.time()
            
        # MediaPipe hand landmark indices
        joints = []
        for landmark in landmarks:
            joints.append(Joint3D(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                confidence=getattr(landmark, 'visibility', 1.0)
            ))
        
        return cls(
            wrist=joints[0],
            thumb_cmc=joints[1],
            thumb_mcp=joints[2],
            thumb_ip=joints[3],
            thumb_tip=joints[4],
            index_mcp=joints[5],
            index_pip=joints[6],
            index_dip=joints[7],
            index_tip=joints[8],
            middle_mcp=joints[9],
            middle_pip=joints[10],
            middle_dip=joints[11],
            middle_tip=joints[12],
            ring_mcp=joints[13],
            ring_pip=joints[14],
            ring_dip=joints[15],
            ring_tip=joints[16],
            pinky_mcp=joints[17],
            pinky_pip=joints[18],
            pinky_dip=joints[19],
            pinky_tip=joints[20],
            hand_type=hand_type,
            timestamp=timestamp
        )
    
    def get_grip_strength(self) -> float:
        """Calculate grip strength based on finger positions."""
        # Distance from thumb tip to index tip (normalized)
        thumb_pos = (self.thumb_tip.x, self.thumb_tip.y, self.thumb_tip.z)
        index_pos = (self.index_tip.x, self.index_tip.y, self.index_tip.z)
        
        distance = sum((a - b) ** 2 for a, b in zip(thumb_pos, index_pos)) ** 0.5
        
        # Convert distance to grip strength (0.0 = closed, 1.0 = open)
        # This is a simplified calculation - adjust based on calibration
        max_distance = 0.15  # Maximum expected distance
        grip_strength = min(distance / max_distance, 1.0)
        
        return grip_strength
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'hand_type': self.hand_type,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'landmarks': {
                'wrist': self.wrist.to_dict(),
                'thumb_cmc': self.thumb_cmc.to_dict(),
                'thumb_mcp': self.thumb_mcp.to_dict(),
                'thumb_ip': self.thumb_ip.to_dict(),
                'thumb_tip': self.thumb_tip.to_dict(),
                'index_mcp': self.index_mcp.to_dict(),
                'index_pip': self.index_pip.to_dict(),
                'index_dip': self.index_dip.to_dict(),
                'index_tip': self.index_tip.to_dict(),
                'middle_mcp': self.middle_mcp.to_dict(),
                'middle_pip': self.middle_pip.to_dict(),
                'middle_dip': self.middle_dip.to_dict(),
                'middle_tip': self.middle_tip.to_dict(),
                'ring_mcp': self.ring_mcp.to_dict(),
                'ring_pip': self.ring_pip.to_dict(),
                'ring_dip': self.ring_dip.to_dict(),
                'ring_tip': self.ring_tip.to_dict(),
                'pinky_mcp': self.pinky_mcp.to_dict(),
                'pinky_pip': self.pinky_pip.to_dict(),
                'pinky_dip': self.pinky_dip.to_dict(),
                'pinky_tip': self.pinky_tip.to_dict(),
            },
            'metadata': self.metadata
        }


@dataclass
class BodySkeleton:
    """Represents a body skeleton with key pose landmarks."""
    
    # Head
    nose: Joint3D
    left_eye: Joint3D
    right_eye: Joint3D
    left_ear: Joint3D
    right_ear: Joint3D
    
    # Torso
    left_shoulder: Joint3D
    right_shoulder: Joint3D
    left_hip: Joint3D
    right_hip: Joint3D
    
    # Arms
    left_elbow: Joint3D
    right_elbow: Joint3D
    left_wrist: Joint3D
    right_wrist: Joint3D
    
    # Legs
    left_knee: Joint3D
    right_knee: Joint3D
    left_ankle: Joint3D
    right_ankle: Joint3D
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'landmarks': {
                'nose': self.nose.to_dict(),
                'left_eye': self.left_eye.to_dict(),
                'right_eye': self.right_eye.to_dict(),
                'left_ear': self.left_ear.to_dict(),
                'right_ear': self.right_ear.to_dict(),
                'left_shoulder': self.left_shoulder.to_dict(),
                'right_shoulder': self.right_shoulder.to_dict(),
                'left_hip': self.left_hip.to_dict(),
                'right_hip': self.right_hip.to_dict(),
                'left_elbow': self.left_elbow.to_dict(),
                'right_elbow': self.right_elbow.to_dict(),
                'left_wrist': self.left_wrist.to_dict(),
                'right_wrist': self.right_wrist.to_dict(),
                'left_knee': self.left_knee.to_dict(),
                'right_knee': self.right_knee.to_dict(),
                'left_ankle': self.left_ankle.to_dict(),
                'right_ankle': self.right_ankle.to_dict(),
            },
            'metadata': self.metadata
        }


@dataclass
class RobotPose:
    """Represents robot pose for teleoperation control."""
    
    # Position
    position: Tuple[float, float, float]  # x, y, z
    
    # Orientation (quaternion)
    orientation: Tuple[float, float, float, float]  # w, x, y, z
    
    # Joint states (for articulated robots)
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            'position': {
                'x': self.position[0],
                'y': self.position[1], 
                'z': self.position[2]
            },
            'orientation': {
                'w': self.orientation[0],
                'x': self.orientation[1],
                'y': self.orientation[2],
                'z': self.orientation[3]
            },
            'joint_positions': self.joint_positions,
            'joint_velocities': self.joint_velocities,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class BasicSkeletonMapper:
    """Basic skeleton mapping utilities for teleoperation."""
    
    @staticmethod
    def hand_to_robot_pose(hand_skeleton: HandSkeleton) -> RobotPose:
        """Convert hand skeleton to robot pose for teleoperation."""
        # Use wrist position as base position
        position = (hand_skeleton.wrist.x, hand_skeleton.wrist.y, hand_skeleton.wrist.z)
        
        # Calculate orientation from hand direction
        # This is a simplified mapping - real implementation would use proper transforms
        thumb_to_index = (
            hand_skeleton.index_tip.x - hand_skeleton.thumb_tip.x,
            hand_skeleton.index_tip.y - hand_skeleton.thumb_tip.y,
            hand_skeleton.index_tip.z - hand_skeleton.thumb_tip.z
        )
        
        # Simplified orientation (would need proper quaternion math)
        orientation = (1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        
        # Map grip strength to gripper joint
        grip_strength = hand_skeleton.get_grip_strength()
        joint_positions = {
            'gripper': grip_strength
        }
        
        return RobotPose(
            position=position,
            orientation=orientation,
            joint_positions=joint_positions,
            timestamp=hand_skeleton.timestamp,
            confidence=hand_skeleton.confidence
        )
    
    @staticmethod
    def body_to_robot_pose(body_skeleton: BodySkeleton) -> RobotPose:
        """Convert body skeleton to robot pose for teleoperation."""
        # Use torso center as base position
        torso_x = (body_skeleton.left_shoulder.x + body_skeleton.right_shoulder.x) / 2
        torso_y = (body_skeleton.left_shoulder.y + body_skeleton.right_shoulder.y) / 2
        torso_z = (body_skeleton.left_shoulder.z + body_skeleton.right_shoulder.z) / 2
        
        position = (torso_x, torso_y, torso_z)
        orientation = (1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        
        return RobotPose(
            position=position,
            orientation=orientation,
            timestamp=body_skeleton.timestamp,
            confidence=body_skeleton.confidence
        )
