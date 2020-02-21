#include <bits/stdc++.h>

using namespace std;

vector<string> split_string(string);


// A C++ program to find the count of distinct substring
// of a string using trie data structure
#include <bits/stdc++.h>
#define MAX_CHAR 26
using namespace std;

// A Suffix Trie (A Trie of all suffixes) Node
class SuffixTrieNode
{
public:
    SuffixTrieNode *children[MAX_CHAR];
    SuffixTrieNode() // Constructor
    {
        // Initialize all child pointers as NULL
        for (int i = 0; i < MAX_CHAR; i++)
          children[i] = NULL;
    }

    // A recursive function to insert a suffix of the s
    // in subtree rooted with this node
    void insertSuffix(string suffix);
};

// A Trie of all suffixes
class SuffixTrie
{
    SuffixTrieNode *root;
    int _countNodesInTrie(SuffixTrieNode *);
public:
    // Constructor (Builds a trie of suffies of the given text)
    SuffixTrie(string s)
    {
        root = new SuffixTrieNode();

        // Consider all suffixes of given string and insert
        // them into the Suffix Trie using recursive function
        // insertSuffix() in SuffixTrieNode class
        for (int i = 0; i < s.length(); i++)
            root->insertSuffix(s.substr(i));
    }

    //  method to count total nodes in suffix trie
    int countNodesInTrie() { return _countNodesInTrie(root); }
};

// A recursive function to insert a suffix of the s in
// subtree rooted with this node
void SuffixTrieNode::insertSuffix(string s)
{
    // If string has more characters
    if (s.length() > 0)
    {
        // Find the first character and convert it
        // into 0-25 range.
        char cIndex = s.at(0) - 'a';

        // If there is no edge for this character,
        // add a new edge
        if (children[cIndex] == NULL)
            children[cIndex] = new SuffixTrieNode();

        // Recur for next suffix
        children[cIndex]->insertSuffix(s.substr(1));
    }
}

// A recursive function to count nodes in trie
int SuffixTrie::_countNodesInTrie(SuffixTrieNode* node)
{
    // If all characters of pattern have been processed,
    if (node == NULL)
        return 0;

    int count = 0;
    for (int i = 0; i < MAX_CHAR; i++)
    {
        // if children is not NULL then find count
        // of all nodes in this subtrie
        if (node->children[i] != NULL)
            count += _countNodesInTrie(node->children[i]);
    }

    // return count of nodes of subtrie and plus
    // 1 because of node's own count
    return (1 + count);
}

// Returns count of distinct substrings of str
int countDistinctSubstring(string str)
{
    // Construct a Trie of all suffixes
    SuffixTrie sTrie(str);

    // Return count of nodes in Trie of Suffixes
    return sTrie.countNodesInTrie();
}


/*
 * Complete the countSubstrings function below.
 */
vector<int> countSubstrings(string s, vector<vector<int>> queries) {
    /*
     * Write your code here.
     */
     vector<int> rst(queries.size());
     for (int idx = 0; idx < queries.size(); idx++) {
         printf("Loop %d\n", idx);
         rst[idx] = countDistinctSubstring(std::string(&s[queries[idx][0]], &s[queries[idx][1]+1])) - 1;
     }
    return rst;
}

int main()
{
    ofstream fout(getenv("OUTPUT_PATH"));

    string nq_temp;
    getline(cin, nq_temp);

    vector<string> nq = split_string(nq_temp);

    int n = stoi(nq[0]);

    int q = stoi(nq[1]);

    string s;
    getline(cin, s);

    vector<vector<int>> queries(q);
    for (int queries_row_itr = 0; queries_row_itr < q; queries_row_itr++) {
        queries[queries_row_itr].resize(2);

        for (int queries_column_itr = 0; queries_column_itr < 2; queries_column_itr++) {
            cin >> queries[queries_row_itr][queries_column_itr];
        }

        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    vector<int> result = countSubstrings(s, queries);

    for (int result_itr = 0; result_itr < result.size(); result_itr++) {
        fout << result[result_itr];

        if (result_itr != result.size() - 1) {
            fout << "\n";
        }
    }

    fout << "\n";

    fout.close();

    return 0;
}

vector<string> split_string(string input_string) {
    string::iterator new_end = unique(input_string.begin(), input_string.end(), [] (const char &x, const char &y) {
        return x == y and x == ' ';
    });

    input_string.erase(new_end, input_string.end());

    while (input_string[input_string.length() - 1] == ' ') {
        input_string.pop_back();
    }

    vector<string> splits;
    char delimiter = ' ';

    size_t i = 0;
    size_t pos = input_string.find(delimiter);

    while (pos != string::npos) {
        splits.push_back(input_string.substr(i, pos - i));

        i = pos + 1;
        pos = input_string.find(delimiter, i);
    }

    splits.push_back(input_string.substr(i, min(pos, input_string.length()) - i + 1));

    return splits;
}
