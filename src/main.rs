use rand::distributions::{Distribution, Uniform};
use rand::rngs::ThreadRng;
use rand::Rng;
use std::env;
use std::fs;
use std::time::Instant;

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

fn calculate_energy(graph: &Graph, solution: &Vec<i64>) -> i64 {
    let mut energy = 0;
    for i in 0..graph.n {
        for (j, w) in &graph.edges[i] {
            if i < *j {
                energy += w * (1 - solution[i] * solution[*j]) / 2;
            }
        }
    }
    energy
}

fn calculate_energy_diff(graph: &Graph, solution: &Vec<i64>, pos: usize) -> i64 {
    solution[pos]
        * graph.edges[pos]
            .iter()
            .map(|(j, w)| *w * solution[*j])
            .sum::<i64>()
}

#[test]
fn energy_diff_test() {
    let data = "data/G1";
    let graph = load_problem(data);

    let mut solution = vec![0i64; graph.n];
    for i in 0..graph.n {
        solution[i] = if i % 2 == 0 { 1 } else { -1 };
    }

    let mut rng = rand::thread_rng();
    let uniform_rand = Uniform::from(0..graph.n);

    for _ in 0..10000 {
        let pos = uniform_rand.sample(&mut rng);
        let energy_before = calculate_energy(&graph, &solution);
        let energy_diff = calculate_energy_diff(&graph, &solution, pos);
        solution[pos] *= -1;
        let energy_after = calculate_energy(&graph, &solution);
        assert_eq!(energy_before + energy_diff, energy_after);
    }
}

fn accept(energy_diff: i64, progress: f64, rng: &mut ThreadRng) -> bool {
    if energy_diff >= 0 {
        true
    } else {
        let prob = (energy_diff as f64 * progress * 1e-4).exp();
        rng.gen::<f64>() <= prob
    }
}

fn simulated_annealing(graph: &Graph, timeout: u128) -> (i64, Vec<i64>) {
    let start_time = Instant::now();

    // initial solution
    let mut solution = vec![0i64; graph.n];
    for i in 0..graph.n {
        solution[i] = if i % 2 == 0 { 1 } else { -1 };
    }

    let mut energy = calculate_energy(&graph, &solution);

    let mut best_energy = energy;
    let mut best_solution = solution.clone();

    let mut counter = 0u64;
    let check_frequency = 8192;
    let mut progress = 0f64;

    let mut rng = rand::thread_rng();
    let uniform_rand = Uniform::from(0..graph.n);

    loop {
        // change state
        let pos = uniform_rand.sample(&mut rng);
        let energy_diff = calculate_energy_diff(&graph, &solution, pos);
        if accept(energy_diff, progress, &mut rng) {
            solution[pos] *= -1;
            energy += energy_diff;
        }

        if best_energy < energy {
            best_energy = energy;
            best_solution = solution.clone();
        }

        counter += 1;
        if counter % check_frequency == 0 {
            let end = start_time.elapsed().as_millis();
            if end > timeout {
                break;
            }
            progress = (end as f64) / (timeout as f64);
        }
    }
    println!("{}", counter);
    (best_energy, best_solution)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let graph = load_problem(&args[1].to_string());

    // max-cut
    // \sum_ij G_ij (1 - x_i x_j)
    // x_i \in {-1, 1}

    let timeout = 5000; // [ms]
    let best_energy, best_solution = simulated_annealing(&graph, timeout);

    println!("energy = {}", best_energy);
    // println!("solution = {:?}", best_solution);

    let last_energy = calculate_energy(&graph, &best_solution);
    println!("energy = {}", last_energy);
}
