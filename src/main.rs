use std::env;
use std::fs;

struct Graph {
    edges: Vec<Vec<(usize, i64)>>,
    n: usize,
}

impl Graph {
    fn new(n: usize) -> Self {
        Graph {
            edges: vec![Vec::new(); n],
            n: n,
        }
    }

    fn connect(&mut self, from: usize, to: usize, cost: i64) {
        self.edges[from].push((to, cost));
        self.edges[to].push((from, cost));
    }
}

fn load_problem(path: &str) -> Graph {
    let text = fs::read_to_string(path).unwrap();
    let content: Vec<&str> = text
        .split('\n')
        .filter(|line| line.trim().len() > 0)
        .collect();

    let token: Vec<&str> = content[0].split_whitespace().collect();
    let node_size: usize = token[0].parse().unwrap();
    let edge_size: usize = token[1].parse().unwrap();

    assert!(edge_size + 1usize == content.len());

    let mut graph = Graph::new(node_size);

    for line in &content[1..] {
        let token: Vec<&str> = line.split_whitespace().collect();

        let from: usize = token[0].parse().unwrap();
        let to: usize = token[1].parse().unwrap();
        let cost: i64 = token[2].parse().unwrap();

        graph.connect(from - 1usize, to - 1usize, cost);
    }
    graph
}

#[test]
fn load_data() {
    let data = "data/G1";
    let graph = load_problem(data);
    assert!(graph.n == 800);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let graph = load_problem(&args[1].to_string());
    println!("{}", graph.n);
}
