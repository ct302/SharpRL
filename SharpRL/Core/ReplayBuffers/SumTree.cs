using System;

namespace SharpRL.Core.ReplayBuffers
{
    /// <summary>
    /// Sum Tree Data Structure for Prioritized Experience Replay
    /// 
    /// NFL ANALOGY:
    /// Think of this like a draft board where teams are ranked by priority (need).
    /// The Sum Tree lets us quickly:
    /// 1. Sample players based on team needs (O(log n))
    /// 2. Update priorities when needs change (O(log n))
    /// 3. Get total "draft capital" instantly (O(1))
    /// </summary>
    public class SumTree
    {
        private readonly float[] tree;
        private readonly int capacity;
        private int writeIndex;
        private int count;

        public SumTree(int capacity)
        {
            this.capacity = capacity;
            tree = new float[2 * capacity];
            writeIndex = 0;
            count = 0;
        }

        public float Total => tree[1];
        public int Count => count;
        public int Capacity => capacity;

        public int Add(float priority)
        {
            if (priority <= 0)
                throw new ArgumentException("Priority must be positive");

            int treeIndex = writeIndex + capacity;
            Update(treeIndex, priority);
            
            int dataIndex = writeIndex;
            writeIndex = (writeIndex + 1) % capacity;
            
            if (count < capacity)
                count++;
            
            return dataIndex;
        }

        public void UpdatePriority(int dataIndex, float priority)
        {
            if (priority <= 0)
                throw new ArgumentException("Priority must be positive");
            
            int treeIndex = dataIndex + capacity;
            Update(treeIndex, priority);
        }

        public int Sample(float value)
        {
            if (value < 0 || value >= Total)
                throw new ArgumentOutOfRangeException(nameof(value));

            return Retrieve(1, value);
        }

        public float GetPriority(int dataIndex)
        {
            int treeIndex = dataIndex + capacity;
            return tree[treeIndex];
        }

        private void Update(int treeIndex, float priority)
        {
            float change = priority - tree[treeIndex];
            tree[treeIndex] = priority;
            
            while (treeIndex > 1)
            {
                treeIndex /= 2;
                tree[treeIndex] += change;
            }
        }

        private int Retrieve(int index, float value)
        {
            int leftChild = 2 * index;
            int rightChild = leftChild + 1;

            if (leftChild >= tree.Length)
                return index - capacity;

            if (value <= tree[leftChild])
                return Retrieve(leftChild, value);
            else
                return Retrieve(rightChild, value - tree[leftChild]);
        }
    }
}
