
import streamlit as st
import pandas as pd
import numpy as np
import random
import io
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Penjadwalan Kuliah - GA")

st.title("Penjadwalan Kuliah Otomatis — Algoritma Genetika")
st.markdown("Aplikasi ini dibuat sesuai laporan proyek: 5 mata kuliah, 3 dosen, 3 ruang, 5 slot waktu per minggu.")

# --- Dataset (can upload to replace) ---
st.sidebar.header("Data & Parameter")
uploaded = st.sidebar.file_uploader("Unggah CSV dataset (CourseID,CourseName,Dosen,Durasi,PreferensiHari)", type=["csv"])
if uploaded is None:
    df = pd.read_csv("dataset.csv")
else:
    df = pd.read_csv(uploaded)

st.sidebar.markdown("**Dataset preview**")
st.sidebar.dataframe(df)

# --- Parameters ---
pop_size = st.sidebar.number_input("Ukuran populasi", value=50, min_value=10, max_value=500, step=10)
generations = st.sidebar.number_input("Jumlah generasi", value=100, min_value=10, max_value=2000, step=10)
crossover_rate = st.sidebar.slider("Tingkat crossover", 0.0, 1.0, 0.8)
mutation_rate = st.sidebar.slider("Tingkat mutasi", 0.0, 1.0, 0.1)
random_seed = st.sidebar.number_input("Seed (0 untuk acak)", value=0)
if random_seed != 0:
    random.seed(int(random_seed))
    np.random.seed(int(random_seed))

# Problem constants based on report
COURSES = df.to_dict('records')
NUM_COURSES = len(COURSES)
ROOMS = ["Ruang A", "Ruang B", "Ruang C"]
NUM_ROOMS = len(ROOMS)
# There are 5 time slots per week (slot indices 0..4) — could represent day/time combination
SLOTS = ["Slot 1 (Senin)", "Slot 2 (Selasa)", "Slot 3 (Rabu)", "Slot 4 (Kamis)", "Slot 5 (Jumat)"]
NUM_SLOTS = len(SLOTS)

# Represent a chromosome as a list of (slot_index, room_index) for each course (length = NUM_COURSES)

def random_chromosome():
    return [(random.randrange(NUM_SLOTS), random.randrange(NUM_ROOMS)) for _ in range(NUM_COURSES)]

def initialize_population(n):
    return [random_chromosome() for _ in range(n)]

def fitness(chrom):
    # Lower penalty = better, but we'll return higher-is-better fitness score.
    penalty = 0
    # 1) Conflicts: same slot and same room -> hard penalty
    used_room_slot = {}
    used_dosen_slot = {}
    for i, gene in enumerate(chrom):
        slot, room = gene
        course = COURSES[i]
        dosen = course["Dosen"]
        # room-slot conflict
        key_room = (room, slot)
        if key_room in used_room_slot:
            penalty += 100  # severe penalty for room clash
        else:
            used_room_slot[key_room] = i
        # dosen-slot conflict
        key_dosen = (dosen, slot)
        if key_dosen in used_dosen_slot:
            penalty += 100  # severe penalty for lecturer clash
        else:
            used_dosen_slot[key_dosen] = i
        # preference day penalty (preferensi hari)
        pref = str(course.get("PreferensiHari","")).lower()
        # Check if preferred day contains the chosen slot day
        slot_name = SLOTS[slot].lower()
        if pref and (slot_name.split()[2] not in pref):
            # small penalty if not preferred day
            penalty += 5 * int(course.get("Durasi",1))
    # 2) Soft objective: spread usage (balance) - reward lower variance of room usage
    room_counts = [0]*NUM_ROOMS
    for _, room in chrom:
        room_counts[room] += 1
    variance = np.var(room_counts)
    penalty += variance*2  # smaller penalty for imbalance
    # Convert to fitness where higher is better
    base = 1000.0
    fit = base - penalty
    return fit

def tournament_selection(pop, fits, k=3):
    selected = random.sample(list(range(len(pop))), k)
    best = max(selected, key=lambda i: fits[i])
    return pop[best]

def crossover(a, b):
    # one-point crossover
    if len(a) <=1: return a.copy(), b.copy()
    point = random.randrange(1, len(a))
    child1 = a[:point] + b[point:]
    child2 = b[:point] + a[point:]
    return child1, child2

def mutate(chrom):
    # mutate by changing slot or room of a random course
    idx = random.randrange(len(chrom))
    if random.random() < 0.5:
        # change slot
        chrom[idx] = (random.randrange(NUM_SLOTS), chrom[idx][1])
    else:
        # change room
        chrom[idx] = (chrom[idx][0], random.randrange(NUM_ROOMS))

# --- GA main loop ---
run_ga = st.sidebar.button("Jalankan GA")
best_history = []
best_chrom = None
best_fit = -1e9

if run_ga:
    pop = initialize_population(pop_size)
    fits = [fitness(ind) for ind in pop]
    best_history = []
    for gen in range(int(generations)):
        new_pop = []
        # Elitism: carry best 2
        sorted_idx = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
        elite_count = max(1, int(0.02*pop_size))
        for ei in sorted_idx[:elite_count]:
            new_pop.append(pop[ei])
        while len(new_pop) < pop_size:
            parent1 = tournament_selection(pop, fits)
            parent2 = tournament_selection(pop, fits)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            # mutation
            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate and len(new_pop)+1 < pop_size:
                mutate(child2)
            new_pop.append(child1)
            if len(new_pop) < pop_size:
                new_pop.append(child2)
        pop = new_pop[:pop_size]
        fits = [fitness(ind) for ind in pop]
        gen_best_idx = int(np.argmax(fits))
        gen_best_fit = fits[gen_best_idx]
        gen_best = pop[gen_best_idx]
        best_history.append(gen_best_fit)
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_chrom = gen_best.copy()
    st.success(f"GA selesai — best fitness: {best_fit:.2f}")
    # Show fitness evolution
    fig, ax = plt.subplots()
    ax.plot(best_history)
    ax.set_xlabel("Generasi")
    ax.set_ylabel("Fitness terbaik")
    ax.set_title("Evolusi Fitness per Generasi")
    st.pyplot(fig)
    # Build schedule table
    schedule = []
    for i, gene in enumerate(best_chrom):
        slot, room = gene
        course = COURSES[i]
        schedule.append({
            "CourseID": course["CourseID"],
            "CourseName": course["CourseName"],
            "Dosen": course["Dosen"],
            "Durasi": course["Durasi"],
            "Slot": SLOTS[slot],
            "Ruang": ROOMS[room]
        })
    sched_df = pd.DataFrame(schedule)
    st.subheader("Hasil Jadwal")
    st.dataframe(sched_df)
    # Download CSV
    csv = sched_df.to_csv(index=False).encode('utf-8')
    st.download_button("Unduh Jadwal (CSV)", csv, "jadwal_hasil.csv", "text/csv")
else:
    st.info("Tekan 'Jalankan GA' di sidebar untuk memulai proses penjadwalan. Kamu bisa mengganti dataset lewat uploader di sidebar.")

st.markdown("---")
st.markdown("**Catatan implementasi**: Algoritma di atas memakai representasi sederhana (slot, room) per mata kuliah. Untuk durasi >1 jam, atau slot multi-hour, perlu penyesuaian representasi jadwal (mis. menggunakan blok slot berurutan).")
