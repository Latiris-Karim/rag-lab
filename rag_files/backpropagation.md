Backpropagation oder auch Backpropagation of Error bzw. Fehlerrückführung[1] bzw. Rückpropagierung ist ein verbreitetes Verfahren zum Einlernen künstlicher neuronaler Netze. Es gehört in der einfachen Form zur Gruppe der überwachten Lernverfahren und wird als Verallgemeinerung der Delta-Regel auf mehrschichtige Netze angewandt. Die Rückwärtspropagierung ist ein Spezialfall eines allgemeinen Gradientenverfahrens in der Optimierung, basierend auf dem mittleren quadratischen Fehler.

Funktionsweise
Ein künstliches neuronales Netz erhält Eingabedaten und bildet diese auf einen Ausgabewert ab. Die numerischen Eingabedaten werden in jeder Schicht des Netzes mit einer Matrix von Gewichten multipliziert, welche die Parameter des Netzes sind.

Beim Training des neuronalen Netzes wird die Abweichung zwischen dem Ausgabewert des Netzes und dem Zielwert des externen Lehrers (im einfachen Fall) berechnet und dieser Fehler rückwärts durch alle Schichten des Netzes propagiert (Backpropagation). Durch Backpropagation lässt sich ermitteln, welche Pfade den größten Einfluss auf den Fehler haben; sie ermöglicht es, Gewichte von Verbindungen zu stärken oder zu schwächen, um den Fehler zu verringern und eine gewünschte Vorhersage zu erreichen.[2]

Im weiterentwickelten Fall (etwa bei der Architektur von AlphaGo) kann auch der menschliche externe Lehrer fehlen und der Zielwert aus anderen Teilen der Architektur geliefert werden.

Algorithmus
Der Backpropagation-Algorithmus läuft in folgenden Phasen:

Ein Eingabemuster wird angelegt und vorwärts durch das Netz propagiert.
Die Ausgabe des Netzes wird mit der gewünschten Ausgabe verglichen. Die Differenz der beiden Werte wird als Fehler des Netzes erachtet.
Der Fehler wird nun wieder über die Ausgabe- zur Eingabeschicht zurück propagiert. Dabei werden die Gewichtungen der Neuronenverbindungen abhängig von ihrem Einfluss auf den Fehler geändert. Dies garantiert bei einem erneuten Anlegen der Eingabe eine Annäherung an die gewünschte Ausgabe.
Der Name des Algorithmus ergibt sich aus dem Zurückpropagieren des Fehlers (engl. error back-propagation).

Fehlerminimierung
Beim Lernproblem wird für beliebige Netze eine möglichst zuverlässige Abbildung von gegebenen Eingabe- auf gegebene Ausgabevektoren angestrebt. Dazu wird die Qualität der Abbildung durch eine Fehlerfunktion beschrieben, die hier durch den quadratischen Fehler definiert wird:

Das Ziel ist nun die Minimierung der Fehlerfunktion, wobei im Allgemeinen lediglich ein lokales Minimum gefunden wird.

Das Einlernen eines künstlichen neuronalen Netzes erfolgt bei dem Backpropagation-Verfahren durch die Änderung der Gewichte, da die Ausgabe des Netzes – außer von der Aktivierungsfunktion – direkt von ihnen abhängt.

Geschichte
Nach verschiedenen Quellen[3][4][5][6] wurden die Grundlagen des Verfahrens im Kontext der Steuerungstheorie hergeleitet durch Prinzipien dynamischer Programmierung, und zwar durch Henry J. Kelley im Jahre 1960[7] und Arthur E. Bryson im Jahre 1961.[8] 1962 publizierte Stuart Dreyfus eine einfachere Herleitung durch die Kettenregel.[9] Vladimir Vapnik zitiert einen Artikel aus dem Jahre 1963[10] in seinem Buch über Support Vector Machines. 1969 beschrieben Bryson und Yu-Chi Ho das Verfahren als mehrstufige Optimierung dynamischer Systeme.[11][12]

Seppo Linnainmaa publizierte im Jahre 1970 schließlich die allgemeine Methode für automatisches Differenzieren (AD) diskreter Netzwerke verschachtelter differenzierbarer Funktionen.[13][14] Dies ist die moderne Variante des Backpropagation-Verfahrens, welche auch bei dünner Vernetzung effizient ist.[15][16][5][6]

1973 verwendete Stuart Dreyfus Backpropagation, um Parameter von Steuersystemen proportional zu ihren Fehlergradienten zu adjustieren.[17] Paul Werbos erwähnte 1974 die Möglichkeit, dieses Prinzip auf künstliche neuronale Netze anzuwenden,[18] und im Jahre 1982 tat er dies auf die heute weit verbreitete Art und Weise.[19][6] 1986 zeigten David E. Rumelhart, Geoffrey E. Hinton und Ronald J. Williams durch Experimente, dass diese Methode zu nützlichen internen Repräsentationen von Eingabedaten in tieferen Schichten neuronaler Netze führen kann, was die Grundlage von Deep Learning ist.[20] Eric A. Wan war 1993 der erste,[5] der einen internationalen Mustererkennungswettbewerb durch Backpropagation gewann.[21]

Modifizierung der Gewichte
Die Variable 
δ
j
{\displaystyle \delta _{j}} geht dabei auf die Unterscheidung der Neuronen ein: Liegt das Neuron in einer verdeckten Schicht, so wird seine Gewichtung abhängig von dem Fehler geändert, den die nachfolgenden Neuronen erzeugen, welche wiederum ihre Eingaben aus dem betrachteten Neuron beziehen.

{\displaystyle w_{ij}^{\mbox{alt}}})
Das Ziel der Backpropagation ist es, die Ableitung des Fehlers in Bezug auf die Gewichte im Netz zu finden. Wenn nach der Änderung eines Wertes in Bezug auf einen anderen Wert gesucht ist, ist dies eine Ableitung. Für die Berechnung repräsentiert jedes Neuron eine Funktion und jede Kante führt einen Vorgang auf dem angehängten Neuron aus. Man beginnt mit dem Fehlerneuron und bewegt sich jeweils ein Neuron zurück und nimmt die partielle Ableitung des aktuellen Neurons in Bezug auf den Neurons in der vorhergehenden Schicht. Jeder Ausdruck wird an den vorhergehenden Ausdruck gekettet, um den Gesamtwert zu berechnen. Dies ist die Kettenregel.[2]

Kettenregel
Die Vorhersagen sind eine lineare Funktion, gefolgt von einer Sigmoidfunktion. Das Modell und die Verlustfunktion lauten wie folgt:

Partielle Ableitungen können auf dieselbe Weise berechnet werden, wie univariate Ableitungen berechnet würden. Man kann die Kostenfunktion in Bezug auf 
 und 
 erweitern und dann die Ableitungen mithilfe wiederholter Anwendungen der univariaten Kettenregel berechnen:

Kostenfunktionen für das maschinelle Lernen basieren normalerweise auf Matrixberechnungen. Eine Theorie besagt, dass es immer einen Algorithmus zum Berechnen von Ableitungen einer Kostenfunktion gibt, der höchstens ein paar Mal so viele Operationen verwendet wie die Kostenfunktion selbst.

Dazu kann man einen Berechnungsgraphen betrachten, der mit der Berechnung eines Skalars 

Aktivierungsfunktion
Der Backpropagation-Algorithmus sucht nach dem Minimum der Fehlerfunktion unter Verwendung des Gradientenverfahrens. Die Kombination von Gewichten, die die Fehlerfunktion minimiert, wird als Lösung des Lernproblems angesehen. Weil dieses Verfahren die Berechnung des Gradienten der Fehlerfunktion bei jedem Iterationsschritt erfordert, muss die Fehlerfunktion stetig und differenzierbar sein.



Viele andere Arten von Aktivierungsfunktionen wurden vorgeschlagen und der Backpropagation-Algorithmus ist auf alle anwendbar. Eine differenzierbare Aktivierungsfunktion macht die von einem neuronalen Netz berechnete Funktion differenzierbar unter der Annahme, dass die Integralfunktion an jedem Knoten nur die Summe der Eingaben ist, weil das Netz selbst nur zusammengesetzte Funktionen berechnet.[25]

Erweiterungen
Die Wahl der Lernrate 
η
{\displaystyle \eta } ist wichtig für das Verfahren, da

ein zu hoher Wert eine starke Veränderung bewirkt, wodurch das Minimum verfehlt werden kann
ein zu kleiner Wert das Einlernen unnötig verlangsamt.
Verschiedene Optimierungen von Rückwärtspropagierung, z. B. Quickprop, zielen vor allem auf die Beschleunigung der Fehlerminimierung; andere Verbesserungen versuchen vor allem die Zuverlässigkeit zu erhöhen.

Backpropagation mit variabler Lernrate
Um eine Oszillation des Netzes, d. h. alternierende Verbindungsgewichte zu vermeiden, existieren Verfeinerungen des Verfahrens, bei dem mit einer variablen Lernrate gearbeitet wird.

Backpropagation mit Trägheitsterm
Durch die Verwendung eines variablen Trägheitsterms (Momentum) 
α
{\displaystyle \alpha } kann der Gradient und die letzte Änderung gewichtet werden, so dass die Gewichtsanpassung zusätzlich von der vorausgegangenen Änderung abhängt. Ist das Momentum 
α
{\displaystyle \alpha } gleich 0, so hängt die Änderung allein vom Gradienten ab, bei einem Wert von 1 lediglich von der letzten Änderung.

Ähnlich einer Kugel, die einen Berg hinunter rollt und deren aktuelle Geschwindigkeit nicht nur durch die aktuelle Steigung des Berges, sondern auch durch ihre eigene Trägheit bestimmt wird, lässt sich der Backpropagation ein Trägheitsterm hinzufügen:

Durch den Trägheitsterm werden unter anderem Probleme der Backpropagation-Regel in steilen Schluchten und flachen Plateaus vermieden. Da zum Beispiel in flachen Plateaus der Gradient der Fehlerfunktion sehr klein wird, käme es ohne Trägheitsterm unmittelbar zu einem „Abbremsen“ des Gradientenabstiegs, dieses „Abbremsen“ wird durch die Addition des Trägheitsterms verzögert, so dass ein flaches Plateau schneller überwunden werden kann.

Sobald der Fehler des Netzes minimal wird, kann das Einlernen abgeschlossen werden und das mehrschichtige Netz ist nun bereit, die erlernten Muster zu klassifizieren.

