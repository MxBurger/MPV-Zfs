# MPI

Bei MPI handelt es sich um eine Technologie für verteiltes Rechnen mit verteiltem Speicher.
MPI steht für **M**essage **P**assing **I**nterface.

> Im Gegensatz zu Shared-Memory-Systemen (wie bei OpenMP), wo alle Threads auf denselben Speicher zugreifen, hat bei MPI jeder Prozess seinen eigenen Speicher. Die Prozesse kommunizieren durch das explizite Senden und Empfangen von Nachrichten.


## Einleitung

Mit MPI kann man Anwendungen realisieren, die hoch skalierbar sind. MPI ist Hardware- und Plattformunabhängig. Es handelt sich dabei um eine standardisierte API / Spezifikation mit verschiedenen Implementierungen (MPICH, OpenMPI, MS-MPI, …). Bei MPI hat man explizite Kontrolle über die Kommunikation, das zwingt einen zum paralleln Denken. Aufgrund der hohen Verbereitung in HPC wird es oft als "Assembler der Parallelverarbeitung" bezeichnet. MPI kennt die darunterliegende Hardware und optimiert die Kommunikation entsprechend.     

**Shared Memory**
- Begrenzt auf die Anzahl der Kerne/Sockel (8–16 bei Consumer-CPUs, bis 256+ bei HPC-Systemen)
- Probleme: Cache-Kohärenz-Overhead, Speicherbandbreiten-Engpass
- Einfach zu programmieren, aber begrenzt skalierbar

**Distributed-Memory**
- Skaliert auf tausende/millionen Knoten
- Jeder Knoten hat eigenen Speicher
- Explizite Kommunikation erforderlich
- Schwieriger zu programmieren, aber hoch skalierbar

### Das MPI Programmiermodell
![alt text](img/mpi_progmod.png)

- **Node** (Knoten): Eine physische Maschine
- **Prozess**: Eine laufende Programminstanz auf einem Node
- **SPMD**-Modell: Alle Prozesse führen dasselbe Programm auf lokalen Daten aus (Single Program Multiple Data)
- Prozesse kommunizieren über **Nachrichten**
- Ein Knoten kann mehrere MPI-Prozesse ausführen

> Ein Knoten kann mehrere MPI-Prozesse ausführen. Auf Shared-Memory-Knoten nutzt MPI intern Shared-Memory für schnelle Kommunikation. Hybride Ansätze (MPI + OpenMP) nutzen Threads innerhalb eines Knotens und MPI zwischen Knoten.

### Kommunikation in MPI

Die Kommunikation funktioniert über Puffer.
- Man schreibt Nachrichten in Puffer über MPI-Funktionen
- MPI kümmert sich um den Transport
- Der Empfänger liest die Nachricht aus seinem Puffer

Man kann sich das wie Briefkästen vorstellen – man legt einen Brief rein (Send), MPI liefert ihn aus, und der Empfänger holt ihn ab (Receive).

![alt text](img/mpi_buff.png)

### Kompilieren und Ausführen von MPI-Programmen

Zum Kompilieren benötigt man `mpicc` bzw. `mpicxx`. Dabei handelt es sich nicht um eigene Compiler, sondern Wrapper-Skripte. Die rufen im Hintergrund deinen normalen Compiler auf (gcc, clang, etc.) und fügen automatisch hinzu:
- Include-Pfade für die MPI-Header (mpi.h)
- Linker-Flags für die MPI-Bibliothek 

Man könnte also auch manuell kompilieren – ist aber umständlich und fehleranfällig.

Zum Ausführen muss eine MPI-Implementierung installiert sein. 
Diese liefert die Bibliothek `libmpi` mit den eigenlichen MPI-Funktionen und den Launcher `mpiexec`, der die Prozesse startet und die Kommunikation verwaltet.
Mit `mpiexec` kann man MPI-Programme ausführen: 
- Auf einem Rechner: mehrere Prozesse lokal `mpiexec -n 4 my_mpi_program`
- Im Cluster via Hostfile (dieses File listet die verfügbaren Hosts)
- Im Cluster via Job-Scheduler
- Manuell: Hosts direkt auf der Kommandozeile angeben

Wenn man einfach ./my_program ausführt, startet nur ein Prozess. MPI braucht aber mehrere Prozesse, die miteinander kommunizieren können.
`mpiexec -n 4 ./my_program` macht Folgendes:`
1. Startet 4 Instanzen deines Programms
2. Richtet die Kommunikationskanäle zwischen ihnen ein
3. Weist jedem Prozess seinen Rang zu

### Hello World

```cpp
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);           // MPI initialisieren

    int comm_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);  // Anzahl Prozesse
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Eigene Prozess-ID (Rang)

    std::cout << "Hello cruel world from process " << rank << " of " << comm_size << "\n";

    MPI_Finalize();                   // MPI beenden
}
```

**Ausgabe (Reihenfolge kann variieren):**
```
Hello cruel world from process 0 of 4
Hello cruel world from process 3 of 4
Hello cruel world from process 2 of 4
Hello cruel world from process 1 of 4
```

Jeder Prozess bekommt einen eindeutigen Rang (0 bis n-1). Die Ausgabe ist nicht geordnet, weil alle Prozesse parallel laufen und unabhängig auf die Konsole schreiben.

### Hello World mit Nachrichten

```cpp
// ...
if (rank != 0) {
    std::string message = std::format("Seas from process {} of {}", rank, comm_size);
    MPI_Send(message.c_str(), message.length() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
} else {
    std::cout << "Received the following messages:\n";
    const int MAX_LENGTH = 100;
    char buffer[MAX_LENGTH];
    for (int i = 1; i < comm_size; i++) {
        MPI_Recv(buffer, MAX_LENGTH, MPI_CHAR, i, 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
        std::cout << "- " << buffer << std::endl;
    }
}
// ...
```

```
Received the following messages:
- Seas from process 1 of 4
- Seas from process 2 of 4
- Seas from process 3 of 4
```

Hier wird explizit koordiniert: Prozess 0 ist der "Chef" und sammelt Nachrichten von allen anderen ein. Die Reihenfolge ist jetzt garantiert, weil Prozess 0 gezielt nacheinander von Prozess 1, 2, 3, … empfängt.

## Grundlegende MPI Funktionen

|**Funktion**| **Beschreibung** |
| --- | --- |
| `MPI_Init` | Initialisiert die MPI-Umgebung |
| `MPI_Comm_size` | Liefert die Anzahl der Prozesse |
| `MPI_Comm_rank` | Liefert den Rang (ID) des aktuellen Prozesses |
| `MPI_Send` | Sendet eine Nachricht an einen anderen Prozess |
| `MPI_Recv` | Empfängt eine Nachricht von einem anderen Prozess |
| `MPI_Finalize` | Beendet die MPI-Umgebung |

Jedes MPI-Programm **muss** `MPI_Init` **vor** allen anderen MPI-Funktionen und am Ende `MPI_Finalize ` aufrufen. MPI-Funktionen beginnen mit dem Präfix `MPI_` und MPI-Konstanten werden immer groß geschrieben. (z.B. `MPI_COMM_WORLD`, `MPI_CHAR`)

### Punkt-zu-Punkt Nachrichten versenden mit `MPI_Send`

```cpp
int MPI_Send(
    void* buf,           // Pointer auf die zu sendenden Daten
    int count,           // Anzahl der Elemente (nicht Bytes!)
    MPI_Datatype datatype, // Datentyp (z.B. MPI_INT, MPI_CHAR, MPI_DOUBLE)
    int dest,            // Rang des Zielprozesses
    int tag,             // Nachrichten-Tag zur Identifikation
    MPI_Comm comm        // Kommunikator (z.B. MPI_COMM_WORLD)
);
```

Die Parameter lassen sich in zwei Gruppen einteilen:
1. Nachrichtendaten:
 - `buf` - Wo liegen die Daten?
 - `count` - Wie viele Elemente?
 - `datatype` - Welcher Typ?
2. Nachrichten-"Umschlag" (Envelope):
 - `dest` - An wen?
 - `tag` - Welche Art von Nachricht? (zur Unterscheidung verschiedener Nachrichten)
 - `comm` - In welchem Kommunikator?

### Punkt-zu-Punkt Nachrichten empfangen mit `MPI_Recv`

```cpp
int MPI_Recv(
    void* buf,           // Pointer auf den Empfangspuffer (muss groß genug sein!)
    int count,           // maximale Anzahl der zu empfangenden Elemente
    MPI_Datatype datatype, // Datentyp
    int source,          // Rang des Senders (oder MPI_ANY_SOURCE)
    int tag,             // Nachrichten-Tag (oder MPI_ANY_TAG)
    MPI_Comm comm,       // Kommunikator
    MPI_Status* status   // Informationen über empfangene Nachricht
);
```

1. Nachrichtendaten:
- `buf` - Wohin sollen die Daten geschrieben werden?
- `count` - Wie viele Elemente passen maximal in den Puffer?
- `datatype` - Welcher Typ?
2. Nachrichten-"Umschlag":
- `source` - Von wem? (`MPI_ANY_SOURCE` = von irgendwem)
- `tag` - Welcher Tag? (`MPI_ANY_TAG` = beliebiger Tag)
- `comm` - In welchem Kommunikator?
3. Status:
- `status` - Enthält Infos über die tatsächlich empfangene Nachricht (Absender, Tag, Fehlercode)
- Kann `MPI_STATUS_IGNORE` sein, wenn man diese Infos nicht braucht

Der Empfangsbuffer muss **vorher** allokiert werden und groß genug sein. `count` ist hier das Maximum – die tatsächliche Nachricht kann kleiner sein.

### Verhalten von `MPI_Send` und `MPI_Recv`

**Blockierende Semantik**
- `MPI_Send` und `MPI_Recv` sind **blockierende** Operationen
- `MPI_Send` blockiert möglicherweise, bis der Sendepuffer wiederverwendet werden kann
- `MPI_Recv` blockiert, **bis** die Nachricht vollständig empfangen wurde

"Blockierend" bedeutet, dass das Programm an dieser Stelle wartet. Bei `MPI_Recv` ist das klar – man wartet auf die Nachricht. Bei `MPI_Send` ist es komplizierter: MPI garantiert nur, dass man den Sendepuffer danach wieder nutzen darf. Ob die Nachricht schon angekommen ist, ist nicht garantiert. Wenn zwei Prozesse gleichzeitig aufeinander warten (beide rufen zuerst `MPI_Recv` auf), blockieren beide ewig → **Deadlock**

**Reihenfolge-Garantien**
- Nachrichten zwischen **demselben** Sender-Empfänger-Paar werden **nicht überholt** (FIFO)
- Nachrichten von **verschiedenen** Sendern können in beliebiger Reihenfolge ankommen

**Nachrichten-Matching (Zuordnung)**
- Sender und Empfänger im **selben Kommunikator**
- **Tags** müssen passen (oder `MPI_ANY_TAG` beim Empfänger)
- **Datentypen** müssen übereinstimmen

### Kommunikatoren

Ein **Kommunikator** definiert eine Gruppe von Prozessen, die miteinander kommunizieren können.

![alt text](img/mpi_comm_world.png)

`MPI_COMM_WORLD` ist der Standard-Kommunikator und enthält alle Prozesse, man kann aber auch eigene Kommunikatoren erstellen. Prozesse können nur innerhalb desselben Kommunikators kommunizieren, ein Prozess kann aber zu mehreren Kommunikatoren gehören. Die Ränge können in verschiedenen Kommunikatoren unterschiedlich sein. 