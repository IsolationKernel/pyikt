"""
pyikt (c) by Xin Han

pyikt is licensed under a
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <https://creativecommons.org/licenses/by-nc-nd/4.0/>.
"""

import math
import random
import string
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import jit


@jit(nopython=True)
def _fast_dot(x, y):
    """Compute the dot product of x and y using numba.

    Args:
        x: a numpy vector (or list)
        y: a numpy vector (or list)

    Returns:
        x_T.y: dot product result
    """
    return np.dot(x, y)


def _normalized_dot_product(v1, v2, t: float = 1.0):
    """Calculate normalized dot product between two vectors.

    Args:
        v1: First vector
        v2: Second vector
        t: Scaling parameter

    Returns:
        Normalized dot product
    """
    v1_norm = math.sqrt(_fast_dot(v1, v1))
    v2_norm = math.sqrt(_fast_dot(v2, v2))
    return _fast_dot(v1, v2) / (t * v1_norm * v2_norm)


class INODE:
    """Isolation hierarchical clustering node."""

    def __init__(self):
        self.id = "id" + "".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(15)
        )
        self.children: List["INODE"] = []
        self.parent: Optional["INODE"] = None
        self.pts: List[Tuple] = []  # each pt is a tuple of (label, id)
        self.ikv = None  # Isolation kernel value
        self.point_counter: int = 0

    def __lt__(self, other: "INODE") -> bool:
        """An arbitrary way to determine an order when comparing 2 nodes."""
        return self.id < other.id

    def insert(self, p_id, p_label, p_ik, t: float = 200):
        """Insert a new pt into the tree.

        Apply recurse masking and balance rotations where appropriate.

        Args:
            pt: a tuple of numpy array, class label, point id
            delete_node: whether to delete nodes when over capacity
            L: maximum number of leaves in the tree
            t: parameter of isolation kernel

        Returns:
            A pointer to the root.
        """
        pt = (p_label, p_id, p_ik)
        if self.pts is not None and len(self.pts) == 0:
            self.add_pt(pt[:2])
            self.ikv = pt[2]
            return self
        else:
            curr_node = self.root()
            x_ik = pt[2].astype(float)
            curr_node = self._find_closest_leaf(t, curr_node, x_ik)
            new_leaf = curr_node._split_down(pt)
            new_leaf._add_node_value_recursively(pt)
            return new_leaf.root()

    def _find_closest_leaf(self, t, curr_node, x_ik):
        """Find the closest leaf to the given point."""
        while curr_node.is_internal():
            chl_ik = curr_node.children[0].ikv.astype(float)
            chr_ik = curr_node.children[1].ikv.astype(float)

            x_dot_chl = _normalized_dot_product(x_ik, chl_ik, t)
            x_dot_chr = _normalized_dot_product(x_ik, chr_ik, t)

            if x_dot_chl >= x_dot_chr:
                curr_node = curr_node.children[0]
            else:
                curr_node = curr_node.children[1]
        return curr_node

    def delete(self):
        """Delete a point from the tree and reorganize structure.

        Returns:
            Updated root node
        """
        current_node = self.root()
        if not current_node.pts:
            return current_node
        pt_id = current_node.pts[0]
        current_node = self._find_and_delete_point(current_node, pt_id)
        current_node._delete_ancestors_ikv()
        sibling_node = current_node.siblings()[0] if current_node.siblings() else None
        if not sibling_node:
            return current_node  # No siblings, return self
        parent_node = current_node.parent
        if parent_node and parent_node.parent:
            parent_node.parent.children.remove(parent_node)
            parent_node.parent.add_child(sibling_node)
            return self.root()
        else:
            # Parent node is root, make sibling the new root
            sibling_node.parent = None
            return sibling_node

    def _delete_ancestors_ikv(self):
        ancs = self._ancestors()
        for a in ancs:
            if a.ikv is not None and self.ikv is not None:
                a.ikv = a.ikv - self.ikv

    def _find_and_delete_point(self, current_node, pt_id):
        while current_node.is_internal():
            current_node.pts.remove(pt_id)
            # assert (p_id in curr_node.children[0].pts[0]) != (p_id in curr_node.children[1].pts[0]), "Except: Exsiting only in  one subtree, \
            #                                                      Get: %s %s" % (p_id in curr_node.children[0].pts[0],
            #                                                                     p_id in curr_node.children[1].pts[0])
            if pt_id in current_node.children[0].pts:
                current_node = current_node.children[0]
            elif pt_id in current_node.children[1].pts:
                current_node = current_node.children[1]
            else:
                # Point not found in either child
                break
        return current_node

    def _add_ik_value(self):
        """Update isolation kernel value based on children's values.

        Returns:
            Updated node
        """
        if self.children:
            if len(self.children) == 1:
                self.ikv = self.children[0].ikv
            else:
                self.ikv = self.children[0].ikv + self.children[1].ikv
        return self

    def _add_node_value_recursively(self, pt):
        """Update a node's parameters recursively up to the root.

        Returns:
            A pointer to the root.
        """
        current_node = self
        while current_node.parent:
            current_node.parent.add_pt(pt[:2])
            current_node.parent._add_ik_value()
            current_node = current_node.parent
        return current_node

    def add_child(self, new_child: "INODE") -> "INODE":
        """Add a INode as a child of this node.

        Args:
            new_child: a INode to add as a child

        Returns:
            A pointer to self with modifications
        """
        new_child.parent = self
        self.children.append(new_child)
        return self

    def add_pt(self, pt) -> "INODE":
        """Add a data point to this node.

        Args:
            pt: the data point to add

        Returns:
            A pointer to this node
        """
        self.point_counter += 1
        if self.pts is not None:
            self.pts.append(pt)
        return self

    def _split_down(self, pt):
        """Create a new node for pt and a new parent with self and pt as children.

        Args:
            pt: the pt to be added

        Returns:
            A pointer to the new node containing pt
        """
        new_internal_node = INODE()
        if self.pts is not None:
            new_internal_node.pts = self.pts[:]  # Copy points to the new node
        else:
            new_internal_node.pts = None
        new_internal_node.point_counter = self.point_counter

        if self.parent:
            self.parent.add_child(new_internal_node)
            self.parent.children.remove(self)
            new_internal_node.add_child(self)
        else:
            new_internal_node.add_child(self)

        new_leaf_node = INODE()
        new_leaf_node.ikv = pt[2]
        new_leaf_node.add_pt(pt[:2])  # This updates the points counter
        new_internal_node.add_child(new_leaf_node)
        return new_leaf_node

    def purity(self, cluster=None):
        """Compute the purity of this node.

        Args:
            cluster: (optional) str, compute purity with respect to this cluster

        Returns:
            A float representing the purity of this node
        """
        if cluster:
            pts = [p for l in self.leaves() for p in l.pts]
            return (
                float(len([pt for pt in pts if pt[0] == cluster])) / len(pts)
                if pts
                else 0
            )
        else:
            label_to_count = self.class_counts()

        total = sum(label_to_count.values())
        return max(label_to_count.values()) / total if total > 0 else 0

    def clusters(self):
        """Return all clusters (true leaves) in the tree.

        Returns:
            List of leaf nodes
        """
        return self.true_leaves()

    def class_counts(self) -> Dict:
        """Produce a map from label to the # of descendant points with label."""
        label_to_count = defaultdict(float)
        pts = [p for l in self.leaves() for p in l.pts]
        for x in pts:
            l, _ = x
            label_to_count[l] += 1.0
        return label_to_count

    def pure_class(self):
        """If this node has purity 1.0, return its label; else return None."""
        cc = self.class_counts()
        if len(cc) == 1:
            return list(cc.keys())[0]
        else:
            return None

    def siblings(self) -> List["INODE"]:
        """Return a list of my siblings."""
        if self.parent:
            return [child for child in self.parent.children if child != self]
        else:
            return []

    def aunts(self) -> List["INODE"]:
        """Return a list of all of my aunts."""
        if self.parent and self.parent.parent:
            return [
                child for child in self.parent.parent.children if child != self.parent
            ]
        else:
            return []

    def _ancestors(self) -> List["INODE"]:
        """Return all of this nodes ancestors in order to the root."""
        anc = []
        curr = self
        while curr.parent:
            anc.append(curr.parent)
            curr = curr.parent
        return anc

    def depth(self) -> int:
        """Return the number of ancestors on the root to leaf path."""
        return len(self._ancestors())

    def height(self) -> int:
        """Return the height of this node (maximum depth of any leaf)."""
        leaves = self.leaves()
        return max([l.depth() for l in leaves]) if leaves else 0

    def descendants(self) -> List["INODE"]:
        """Return all descendants of the current node."""
        d = []
        queue = deque([self])
        while queue:
            n = queue.popleft()
            d.append(n)
            if n.children:
                queue.extend(n.children)
        return d

    def leaves(self) -> List["INODE"]:
        """Return the list of leaves under this node."""
        lvs = []
        queue = deque([self])
        while queue:
            n = queue.popleft()
            if n.children:
                queue.extend(n.children)
            else:
                lvs.append(n)
        return lvs

    def lca(self, other: "INODE") -> "INODE":
        """Compute the lowest common ancestor between this node and other.

        Args:
            other: a node in the tree

        Returns:
            A node that is the lowest common ancestor
        """
        ancestors = set(self._ancestors() + [self])
        curr_node = other
        while curr_node not in ancestors:
            if not curr_node.parent:
                break  # No common ancestor found
            curr_node = curr_node.parent
        return curr_node

    def root(self) -> "INODE":
        """Return the root of the tree."""
        curr_node = self
        while curr_node.parent:
            curr_node = curr_node.parent
        return curr_node

    def is_leaf(self) -> bool:
        """Returns true if self is a leaf, else false."""
        return len(self.children) == 0

    def is_internal(self) -> bool:
        """Returns false if self is a leaf, else true."""
        return not self.is_leaf()
