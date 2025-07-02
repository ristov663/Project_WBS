from pyvis.network import Network

# Иницијализација на мрежата со подобрени поставки
net = Network(
    notebook=False,
    height="800px",
    width="100%",
    bgcolor="#ffffff",
    font_color="#2d3436",
    heading="Ontology Visualization - Public Procurement"
)

# Додавање на јазли со академички стил
net.add_node("Contract",
            shape="box",
            color="#1750AC",
            size=40,
            title="Contract\n- hasDescription: xsd:string\n- hasDate: xsd:dateTime\n- hasAmount: xsd:float",
            font={"size": 14, "face": "Arial"})

net.add_node("Institution",
            shape="ellipse",
            color="#3373C4",
            size=30,
            title="Institution Class",
            font={"size": 14})

net.add_node("Supplier",
            shape="ellipse",
            color="#3373C4",
            size=30,
            title="Supplier Class",
            font={"size": 14})

# Податочни својства со различна форма
data_properties = {
    "hasDescription": {"type": "xsd:string", "color": "#86CEFA"},
    "hasDate": {"type": "xsd:dateTime", "color": "#86CEFA"},
    "hasAmount": {"type": "xsd:float", "color": "#86CEFA"}
}

for prop, meta in data_properties.items():
    net.add_node(prop,
                shape="diamond",
                color=meta["color"],
                size=25,
                title=f"{prop}\nRange: {meta['type']}",
                font={"size": 12})

# Додавање на врски со подобрени стилови
relationships = [
    ("Contract", "Institution", "hasInstitution", "#3373C4", "dash"),
    ("Contract", "Supplier", "hasSupplier", "#3373C4", "dash"),
    ("Contract", "hasDescription", "", "#86CEFA", "dot"),
    ("Contract", "hasDate", "", "#86CEFA", "dot"),
    ("Contract", "hasAmount", "", "#86CEFA", "dot")
]

for rel in relationships:
    net.add_edge(rel[0], rel[1],
                title=rel[2],
                color=rel[3],
                width=2,
                dashes=rel[4] == "dash",
                arrowStrikethrough=False,
                label=rel[2],
                font={"size": 12, "align": "middle"})

# Конфигурација за физика на мрежата
net.set_options("""
{
  "physics": {
    "hierarchicalRepulsion": {
      "centralGravity": 0,
      "springLength": 150,
      "nodeDistance": 120
    },
    "minVelocity": 0.75,
    "solver": "hierarchicalRepulsion"
  },
  "interaction": {
    "hover": true
  }
}
""")

# Зачувување и приказ
net.save_graph("ontology3.html")
