// decision_tree_sfml.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <SFML/Graphics.hpp>   // link with -lsfml-graphics -lsfml-window -lsfml-system
#include <cfloat> // For FLT_MAX
#include <sstream>

using namespace std;

// ————————————————————————————————————————————————————————————————————————————————
// DataSheet: reads a CSV (comma-delimited) into a 2D vector<string> and computes overall entropy.
// ————————————————————————————————————————————————————————————————————————————————
class DataSheet {
public:
    explicit DataSheet(fstream &file)
        : dataFile{}, entropyOfDatas(0.0)
    {
        readFile(file);
        entropyOfDatas = calculateEntropy();
    }

    void printData() const {
        cout << "There are " << dataFile[0].size() << " attributes and "
             << (dataFile.size() - 1) << " data rows.\n\n";

        for (size_t j = 0; j < dataFile[0].size(); ++j) {
            cout << dataFile[0][j] << (j + 1 == dataFile[0].size() ? "\n" : ", ");
        }
        for (size_t i = 1; i < dataFile.size(); ++i) {
            for (size_t j = 0; j < dataFile[i].size(); ++j) {
                cout << dataFile[i][j] << (j + 1 == dataFile[i].size() ? "\n" : ", ");
            }
        }
        cout << "\n";
    }

    double calculateInformationGain(string_view attributeName) const {
        int attributeIndex = -1;
        int columnCount = static_cast<int>(dataFile[0].size());
        int rowCount = static_cast<int>(dataFile.size());

        for (int i = 0; i < columnCount - 1; ++i) {
            if (dataFile[0][i] == attributeName) {
                attributeIndex = i;
                break;
            }
        }
        if (attributeIndex < 0) {
            cerr << "Attribute not found: " << attributeName << "\n";
            return 0.0;
        }

        unordered_map<string, vector<vector<string>>> partitions;
        for (int i = 1; i < rowCount; ++i) {
            const string &key = dataFile[i][attributeIndex];
            partitions[key].push_back(dataFile[i]);
        }

        double infoGain = entropyOfDatas;

        for (auto const &kv : partitions) {
            const auto &subset = kv.second;
            unordered_map<string,int> labelCount;
            for (auto const &row : subset) {
                const string &lab = row[columnCount - 1];
                labelCount[lab]++;
            }
            double subsetEntropy = 0.0;
            for (auto const &lp : labelCount) {
                double p = static_cast<double>(lp.second) / subset.size();
                subsetEntropy += -p * log2(p);
            }
            infoGain -= (static_cast<double>(subset.size()) / (rowCount - 1)) * subsetEntropy;
        }

        return infoGain;
    }

    const vector<vector<string>>& getData() const {
        return dataFile;
    }

    vector<string> getHeaders() const {
        return dataFile.empty() ? vector<string>{} : dataFile[0];
    }

    double getEntropy() const {
        return entropyOfDatas;
    }

private:
    vector<vector<string>> dataFile;
    double entropyOfDatas;

    double calculateEntropy() {
        unordered_map<string,int> classCount;
        int row = static_cast<int>(dataFile.size());
        int col = static_cast<int>(dataFile[0].size());

        for (int i = 1; i < row; ++i) {
            const string &lab = dataFile[i][col - 1];
            classCount[lab]++;
        }
        double ent = 0.0;
        for (auto const &kv : classCount) {
            double p = static_cast<double>(kv.second) / (row - 1);
            ent += -p * log2(p);
        }
        return ent;
    }

    void splitDelimiter(const string &input, vector<string> &output, char delimiter) {
        output.clear();
        size_t start = 0;
        while (true) {
            size_t pos = input.find(delimiter, start);
            if (pos == string::npos) {
                output.emplace_back(input.substr(start));
                break;
            }
            output.emplace_back(input.substr(start, pos - start));
            start = pos + 1;
        }
    }

    void readFile(fstream &file) {
        string line;
        vector<string> temp;
        while (getline(file, line)) {
            splitDelimiter(line, temp, ',');
            dataFile.emplace_back(temp);
        }
    }
};

// ————————————————————————————————————————————————————————————————————————————————
// TreeNode: each node holds either an attribute (internal node) or a label (leaf).
// We also store an (x,y) for SFML drawing, and for each child, the edge value.
// ————————————————————————————————————————————————————————————————————————————————
struct TreeNode {
    string attribute;     // if internal node
    string label;         // non-empty only if leaf
    unordered_map<string, TreeNode*> children;
    sf::Vector2f position; // for visualization

    TreeNode(const string &attr, const string &lab)
        : attribute(attr), label(lab), position({0,0}) {}
};

// ————————————————————————————————————————————————————————————————————————————————
// DecisionTree: builds recursively on subsets, prints text, and visualizes via SFML.
// ————————————————————————————————————————————————————————————————————————————————
class DecisionTree {
public:
    explicit DecisionTree(DataSheet *data)
        : dataFile(data)
    {
        root = buildTree(dataFile->getData(), dataFile->getHeaders());
    }

    void printTree(TreeNode *node = nullptr, const string &indent = "", const string &edgeValue = "", const string &path = "") const {
        if (!node) {
            if (!root) {
                cout << "Tree is empty.\n";
                return;
            }
            node = root;
        }

        // Build path step-by-step
        string fullPath = path;
        if (!edgeValue.empty()) {
            if (!fullPath.empty()) fullPath += " -> ";
            fullPath += edgeValue;
        }

        // If it's a leaf node
        if (!node->label.empty()) {
            cout << indent << "├── " << fullPath << ": Leaf = " << node->label << "\n";
            return;
        }

        // Internal node (attribute node)
        if (!edgeValue.empty()) {
            cout << indent << "├── " << edgeValue << ": Attribute = " << node->attribute << "\n";
        } else {
            cout << indent << "Attribute = " << node->attribute << "\n";
        }

        // Sort child keys alphabetically
        vector<string> keys;
        for (const auto &kv : node->children) keys.push_back(kv.first);
        sort(keys.begin(), keys.end());

        for (const auto &val : keys) {
            TreeNode *child = node->children.at(val);
            printTree(child, indent + "│   ", val, fullPath);
        }
    }



    // Opens an SFML window and draws the tree. Now with edge value labels.
    void visualize() {
    const int windowWidth = 1200;
    const int windowHeight = 800;
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "Decision Tree");

    sf::Font font;
    if (!font.loadFromFile("DejaVuSans.ttf")) {
        cerr << "ERROR: Could not load font \"DejaVuSans.ttf\". Place it in working directory.\n";
        return;
    }

    float xSpacing = 100.0f;
    float ySpacing = 100.0f;
    float currentX = 50.0f;
    computeNodePositions(root, 0, currentX, xSpacing, ySpacing);

    // Auto-fit view to tree bounds
    sf::FloatRect treeBounds = calculateTreeBounds(root);
    sf::View view(treeBounds);
    view.setViewport(sf::FloatRect(0, 0, 1, 1)); // Full window

    // Interaction variables
    bool dragging = false;
    sf::Vector2i prevMousePos;

    while (window.isOpen()) {
        sf::Event ev;
        while (window.pollEvent(ev)) {
            if (ev.type == sf::Event::Closed) {
                window.close();
            } else if (ev.type == sf::Event::MouseWheelScrolled) {
                float zoomFactor = (ev.mouseWheelScroll.delta > 0) ? 0.9f : 1.1f;
                view.zoom(zoomFactor);
            } else if (ev.type == sf::Event::MouseButtonPressed && ev.mouseButton.button == sf::Mouse::Left) {
                dragging = true;
                prevMousePos = sf::Mouse::getPosition(window);
            } else if (ev.type == sf::Event::MouseButtonReleased && ev.mouseButton.button == sf::Mouse::Left) {
                dragging = false;
            } else if (ev.type == sf::Event::MouseMoved && dragging) {
                sf::Vector2i newMousePos = sf::Mouse::getPosition(window);
                sf::Vector2f delta = window.mapPixelToCoords(prevMousePos) - window.mapPixelToCoords(newMousePos);
                view.move(delta);
                prevMousePos = newMousePos;
            }
        }

        window.clear(sf::Color::White);
        window.setView(view);
        drawTree(window, root, font);
        window.display();
    }
}

// Helper: Compute the bounding rectangle of the entire tree
sf::FloatRect calculateTreeBounds(TreeNode* node) {
    if (!node) return sf::FloatRect(0, 0, 0, 0);

    float minX = node->position.x, maxX = node->position.x;
    float minY = node->position.y, maxY = node->position.y;

    for (const auto& kv : node->children) {
        sf::FloatRect childBounds = calculateTreeBounds(kv.second);
        minX = std::min(minX, childBounds.left);
        maxX = std::max(maxX, childBounds.left + childBounds.width);
        minY = std::min(minY, childBounds.top);
        maxY = std::max(maxY, childBounds.top + childBounds.height);
    }

    return sf::FloatRect(minX - 50, minY - 50, (maxX - minX) + 100, (maxY - minY) + 100);
}

private:
    TreeNode   *root     = nullptr;
    DataSheet  *dataFile = nullptr;

    TreeNode* buildTree(const vector<vector<string>> &data, const vector<string> &headers) {
        int rowCount = static_cast<int>(data.size());
        int colCount = static_cast<int>(headers.size());
        int labelIndex = colCount - 1;

        const string &firstLab = data[1][labelIndex];
        bool allSame = true;
        for (int i = 2; i < rowCount; ++i) {
            if (data[i][labelIndex] != firstLab) {
                allSame = false;
                break;
            }
        }
        if (allSame) {
            return new TreeNode("", firstLab);
        }

        if (colCount <= 2) {
            unordered_map<string,int> freq;
            for (int i = 1; i < rowCount; ++i) {
                freq[data[i][labelIndex]]++;
            }
            string majority;
            int maxC = 0;
            for (auto &kv : freq) {
                if (kv.second > maxC) {
                    maxC = kv.second;
                    majority = kv.first;
                }
            }
            return new TreeNode("", majority);
        }

        int bestIndex = -1;
        double bestGain = -1.0;
        for (int i = 0; i < colCount - 1; ++i) {
            double gain = calculateIG_OnSubset(data, i);
            if (gain > bestGain) {
                bestGain = gain;
                bestIndex = i;
            }
        }
        if (bestIndex < 0) {
            unordered_map<string,int> freq;
            for (int i = 1; i < rowCount; ++i) {
                freq[data[i][labelIndex]]++;
            }
            string majority;
            int maxC = 0;
            for (auto &kv : freq) {
                if (kv.second > maxC) {
                    maxC = kv.second;
                    majority = kv.first;
                }
            }
            return new TreeNode("", majority);
        }

        string bestAttr = headers[bestIndex];
        TreeNode *node = new TreeNode(bestAttr, "");

        unordered_map<string, vector<vector<string>>> partitions;
        for (int i = 1; i < rowCount; ++i) {
            const string &val = data[i][bestIndex];
            vector<string> reducedRow = data[i];
            reducedRow.erase(reducedRow.begin() + bestIndex);
            partitions[val].push_back(move(reducedRow));
        }

        vector<string> newHeaders = headers;
        newHeaders.erase(newHeaders.begin() + bestIndex);

        for (auto &kv : partitions) {
            const string &val = kv.first;
            auto &rowsForVal = kv.second;
            vector<vector<string>> subset;
            subset.push_back(newHeaders);
            for (auto &r : rowsForVal) {
                subset.push_back(move(r));
            }
            node->children[val] = buildTree(subset, newHeaders);
        }
        return node;
    }

    double calculateIG_OnSubset(const vector<vector<string>> &subset, int attrIdx) const {
        int rowCount = static_cast<int>(subset.size());
        int colCount = static_cast<int>(subset[0].size());
        int labelIdx = colCount - 1;

        unordered_map<string,int> labelCount;
        for (int i = 1; i < rowCount; ++i) {
            labelCount[subset[i][labelIdx]]++;
        }
        double totalEnt = 0.0;
        for (auto &kv : labelCount) {
            double p = static_cast<double>(kv.second) / (rowCount - 1);
            totalEnt += -p * log2(p);
        }

        unordered_map<string, vector<vector<string>>> parts;
        for (int i = 1; i < rowCount; ++i) {
            const string &key = subset[i][attrIdx];
            parts[key].push_back(subset[i]);
        }

        double remainder = 0.0;
        for (auto &kv : parts) {
            auto &rowsForVal = kv.second;
            int partSize = static_cast<int>(rowsForVal.size());
            unordered_map<string,int> partLabelCount;
            for (auto &r : rowsForVal) {
                partLabelCount[r[labelIdx]]++;
            }
            double partEnt = 0.0;
            for (auto &lp : partLabelCount) {
                double p2 = static_cast<double>(lp.second) / partSize;
                partEnt += -p2 * log2(p2);
            }
            remainder += (static_cast<double>(partSize) / (rowCount - 1)) * partEnt;
        }

        return totalEnt - remainder;
    }

    void computeNodePositions(TreeNode *node, int depth, float &currentX,
                              float xSpacing, float ySpacing)
    {
        if (!node) return;

        if (!node->label.empty()) {
            node->position.x = currentX;
            node->position.y = depth * ySpacing + 50.0f;
            currentX += xSpacing;
            return;
        }

        vector<string> keys;
        for (auto &kv : node->children)
            keys.push_back(kv.first);
        sort(keys.begin(), keys.end());

        float leftMost = FLT_MAX, rightMost = -1.0f;
        for (auto const &val : keys) {
            TreeNode *child = node->children[val];
            computeNodePositions(child, depth + 1, currentX, xSpacing, ySpacing);
            leftMost = min(leftMost, child->position.x);
            rightMost = max(rightMost, child->position.x);
        }
        node->position.x = (leftMost + rightMost) / 2.0f;
        node->position.y = depth * ySpacing + 50.0f;
    }

    // Improved: Draws nodes, edges, and edge labels (attribute values)
    void drawTree(sf::RenderWindow &win, TreeNode *node, const sf::Font &font) const {
        if (!node) return;

        // Draw lines and value labels from this node to its children
        for (auto const &kv : node->children) {
            TreeNode *child = kv.second;
            if (!child) continue;

            // Draw edge
            sf::Vertex line[] = {
                sf::Vertex(node->position, sf::Color::Black),
                sf::Vertex(child->position, sf::Color::Black)
            };
            win.draw(line, 2, sf::Lines);

            // Draw the value of the edge (attribute value)
            sf::Vector2f mid = (node->position + child->position) / 2.0f;
            sf::Text edgeText;
            edgeText.setFont(font);
            edgeText.setCharacterSize(12);
            edgeText.setFillColor(sf::Color::Blue);
            edgeText.setString(kv.first);
            sf::FloatRect edgeBounds = edgeText.getLocalBounds();
            edgeText.setPosition(mid.x - edgeBounds.width / 2, mid.y - edgeBounds.height / 2);
            win.draw(edgeText);

            drawTree(win, child, font);
        }

        // Draw this node: a small circle and text
        float radius = 20.0f;
        sf::CircleShape circle(radius);
        if (!node->label.empty())
            circle.setFillColor(sf::Color(180,255,180)); // leaf: light green
        else
            circle.setFillColor(sf::Color::White); // internal: white
        circle.setOutlineColor(sf::Color::Black);
        circle.setOutlineThickness(2.0f);
        circle.setPosition(node->position.x - radius, node->position.y - radius);
        win.draw(circle);

        // Prepare text: attribute (internal) or label (leaf)
        sf::Text text;
        text.setFont(font);
        text.setCharacterSize(14);
        text.setFillColor(sf::Color::Black);
        if (!node->label.empty()) {
            text.setString(node->label);
        } else {
            text.setString(node->attribute);
        }
        sf::FloatRect bounds = text.getLocalBounds();
        text.setPosition(
            node->position.x - bounds.width / 2.0f,
            node->position.y - bounds.height / 2.0f - 5.0f
        );
        win.draw(text);
    }
};



using namespace std;

// Function to split a string by a delimiter
vector<string> split(const string& line, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(line);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to read CSV/TXT file with a given delimiter
vector<vector<string>> readTableFromFile(const string& filename, char delimiter) {
    vector<vector<string>> table;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << "\n";
        return table;
    }

    string line;
    while (getline(file, line)) {
        vector<string> row = split(line, delimiter);
        table.push_back(row);
    }

    file.close();
    return table;
}

int main() {
    string filename;
    cout << "Enter CSV or TXT file name to read: ";
    cin >> filename;

    // Convert filename to lowercase to check extension
    string lowerFilename = filename;
    transform(lowerFilename.begin(), lowerFilename.end(), lowerFilename.begin(), ::tolower);

    char delimiter = ',';
    if (lowerFilename.find(".txt") != string::npos) {
        string delimStr;
        cout << "Enter the delimiter character for the TXT file (e.g. , . ; |): ";
        cin >> delimStr;
        if (!delimStr.empty()) {
            delimiter = delimStr[0];
        }
    }

    // Open file
    fstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << "\n";
        return 1;
    }

    // If TXT, set delimiter for DataSheet
    DataSheet data(file);  // This assumes comma as delimiter
    file.close();          // Close after DataSheet reads it

    // Print loaded data

    // Build and print the decision tree
    DecisionTree tree(&data);
    tree.printTree();

    // Visualize the tree
    tree.visualize();

    return 0;
}
